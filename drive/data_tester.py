import numpy as np
import lmdb
import cv2
import pygame
import torch
import sys
import glob
from data_util import YamlConfig, load_config,visualize_birdview, get_birdview
from carla import ColorConverter as cc
from pygame.locals import Color

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
    sys.path.append(glob.glob('../drive')[0])
    sys.path.append('../LearningByCheating')
except IndexError as e:
    pass
from models.birdview import BirdViewPolicyModelSS
# from models.image import ImagePolicyModelSS
from utils.train_utils import one_hot
from torchvision import transforms
birdview_transform = transforms.ToTensor()
BLUE = Color('blue')
RED = Color('red')
PIXELS_PER_METER = 5
N_STEP=5
CROP_SIZE = 320
MAP_SIZE=320
config = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'teacher_args' : {
                'model_path': '/home/moonlab/Documents/LearningByCheating/training/birdview/model-181.th',
                },
            'image_args' : {
                'model_path': '/home/moonlab/Documents/karthik/LearningByCheating/model-249.th',
                }
            }
# image_net = ImagePolicyModelSS(backbone='resnet34').to(config['device'])
# image_net.load_state_dict(torch.load(config['image_args']['model_path']))
# image_net.eval()
# teacher_net = BirdViewPolicyModelSS(backbone='resnet18').to(config['device'])
# teacher_net.load_state_dict(torch.load(config['teacher_args']['model_path']))
# teacher_net.eval()

class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=0.88, fixed_offset=2.5, device='cuda'):
        self._w = w
        self._h = h
        self._img_size = torch.FloatTensor([w,h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        
        self._tran = np.array([0.0,0.0,0.0])
        self._rot  = np.array([0.,0.,0.])
        f = self._w /(2 * np.tan(self._fov * np.pi / 360))
        self._A = np.array([
            [f, 0., self._w/2],
            [0, f, self._h/2],
            [0., 0., 1.]
        ])
        
    def _project_image_xy(self, xy):
        N = len(xy)
        xyz = np.zeros((N,3))
        xyz[:,0] = xy[:,0]
        xyz[:,1] = self._world_y
        xyz[:,2] = xy[:,1]
    
        image_xy, _ = cv2.projectPoints(xyz, self._tran, self._rot, self._A, None)
        image_xy[...,0] = np.clip(image_xy[...,0], 0, self._w)
        image_xy[...,1] = np.clip(image_xy[...,1], 0, self._h)
    
        return image_xy[:,0]
    
    def __call__(self, map_locations):
        if isinstance(map_locations, list):
            map_locations = np.array(map_locations)
        if isinstance(map_locations, torch.Tensor):
            map_locations = map_locations.detach().cpu().numpy()
        
        teacher_locations = (map_locations + 1) * CROP_SIZE / 2
        teacher_locations = np.expand_dims(teacher_locations, axis=0)
        N = teacher_locations.shape[0]
        teacher_locations[:,:,1] = CROP_SIZE - teacher_locations[:,:,1]
        teacher_locations[:,:,0] -= CROP_SIZE/2
        teacher_locations = teacher_locations / PIXELS_PER_METER
        teacher_locations[:,:,1] += self._fixed_offset
        teacher_locations = self._project_image_xy(np.reshape(teacher_locations, (N*N_STEP, 2)))
        teacher_locations = np.reshape(teacher_locations, (N,N_STEP,2))
        return teacher_locations

def world_to_pixel(
        x,y,ox,oy,ori_ox, ori_oy,
        pixels_per_meter=5, offset=(-80,160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x-ox)*pixels_per_meter, (y-oy)*pixels_per_meter

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    pixel_x = 320-pixel_x

    return np.array([pixel_x, pixel_y]) + offset

def crop_birdview(birdview, dx=0, dy=0):
    CROP_SIZE = 192
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview
import sys
args = YamlConfig.from_nested_dicts(load_config('config/hound_config.yaml'))
env = lmdb.open('/home/moonlab/Documents/LearningByCheating/dataset/train/{}'.format(sys.argv[1]))
pygame.init()
pygame.font.init()
display = pygame.display.set_mode(
                    (args.width, args.height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
display.fill((0,0,0))
with env.begin() as txn:
    length = int(txn.get(str('len').encode()))
    for i in range(length-25):
        rgb_left = np.fromstring(txn.get(('rgb_left_%04d'%i).encode()), np.uint8).reshape(160,384,3)
        rgb_right = np.fromstring(txn.get(('rgb_right_%04d'%i).encode()), np.uint8).reshape(160,384,3)
        bird_view = np.fromstring(txn.get(('birdview_%04d'%i).encode()), np.uint8).reshape(320,320,8)
        # removing traffic channels
        bird_view = np.delete(bird_view, [2], axis=-1)
        measurement = np.frombuffer(txn.get(('measurements_%04d'%i).encode()), np.float32)
        display.blit(pygame.surfarray.make_surface(rgb_left.swapaxes(0, 1)), (0, 0))
        display.blit(pygame.surfarray.make_surface(rgb_right.swapaxes(0, 1)), (0, 160))
        display.blit(pygame.surfarray.make_surface(np.zeros((320, 320))), (704, 0))
        birdview = crop_birdview(bird_view)
        bird_view = visualize_birdview(birdview)
        display.blit(pygame.surfarray.make_surface(np.transpose(bird_view, (1, 0, 2))), (384, 0))
        ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear, traffic_light  = measurement
        
        #birdview
        birdview = np.reshape(birdview, (192, 192, 7))
        birdview = birdview_transform(birdview)
        birdview = birdview[None, :].to(config['device'])
        # rgb_left = birdview_transform(rgb_left)
        # rgb_right = birdview_transform(rgb_right)
        # rgb_left = rgb_left[None, :].to(config['device'])
        # rgb_right = rgb_right[None, :].to(config['device'])
        command = one_hot(torch.Tensor([cmd])).to(config['device'])
        speed = np.sqrt(vx**2 + vy**2+vz**2)
        print(float(speed)*18/5)
        speed = torch.Tensor([float(speed)]).to(config['device'])
        # with torch.no_grad():
        #     _teac_locations = teacher_net(birdview, speed, command)
        # coord_converter = CoordConverter()
        # _teac_locations = _teac_locations.squeeze().detach().cpu().numpy()
        # for x, y in (_teac_locations+1) * (0.5*192):
        #     pygame.draw.rect(display, RED, pygame.Rect(int(x+384), int(y), 3, 3))

        gap = 5
        n_step = 5
        ox, oy, oz, ori_ox, ori_oy  = measurement[:5]
        gt_loc = []
        dt = 0
        while dt < 25 :
            index = i + dt
            f_measurement = np.frombuffer(txn.get(("measurements_%04d"%index).encode()), np.float32)
            x, y, z, ori_x, ori_y = f_measurement[:5]
            pixel_y, pixel_x = world_to_pixel(x,y,ox,oy,ori_ox,ori_oy)
            pixel_x = pixel_x - (320-192)//2
            pixel_y = 192 - (320-pixel_y)+70
            gt_loc.append([pixel_x, pixel_y])
            pygame.draw.rect(display, BLUE, pygame.Rect(pixel_x+384, pixel_y, 3, 3))
            dt +=5
        # location = coord_converter(_teac_locations)
        # for x, y in location[0]:
        #     pygame.draw.rect(display, BLUE, pygame.Rect(x, y, 3, 3))

            pygame.display.update()
    pygame.quit()
