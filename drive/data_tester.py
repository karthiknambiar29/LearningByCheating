import numpy as np
import lmdb
import cv2
import pygame
import torch
import math
import os
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
                'model_path': '/home/moonlab/Documents/karthik/LearningByCheating/ckpts/priveleged/model-128.th',
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
    def __init__(self, w=800, h=600, fov=90, world_y=0.88, fixed_offset=0.0, device='cuda'):
        self._w = w
        self._h = h
        self._img_size = torch.FloatTensor([w,h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        
        self._tran = np.array([0.,0.,0.])
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
        xyz[:,1] = 0.88
        xyz[:,2] = xy[:,1]
    
        ROTATION_MATRIX = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ])
        
        image_xy, _ = cv2.projectPoints(xyz, self._tran, self._rot, self._A, None)
        image_xy[...,0] = np.clip(image_xy[...,0], 0, self._w)
        image_xy[...,1] = np.clip(image_xy[...,1], 0, self._h)

        XYZ = np.zeros((xyz.shape[0], 4))
        XYZ[:, 0:-1] = xyz[:, :]
        XYZ[:, 2] +=2.2
        XYZ[:, 0] -= 0.15
        XYZ[:, -1] = np.ones_like(XYZ[:, -1])
        uv = np.matmul(np.matmul(self._A, ROTATION_MATRIX), XYZ.T)
        uv /= uv[-1]
        uv = uv.T
        # print(uv, uv[:, 0:-1])
        return uv[:, 0:-1], #image_xy[:,0]
    
    def __call__(self, map_locations):
        teacher_locations = map_locations.detach().cpu().numpy()
        teacher_locations = (teacher_locations + 1) * 320 / 2
        N = teacher_locations.shape[0]
        teacher_locations[:,:,1] = 320 - teacher_locations[:,:,1]
        teacher_locations[:,:,0] -= 320/2
        teacher_locations = teacher_locations / PIXELS_PER_METER
        # print('teacher_locations', teacher_locations)
        # teacher_locations[:,:,1] += self._fixed_offset
        teacher_locations = self._project_image_xy(np.reshape(teacher_locations, (N*N_STEP, 2)))
        teacher_locations = np.reshape(teacher_locations, (N,N_STEP,2))
        teacher_locations = torch.FloatTensor(teacher_locations)
        return teacher_locations

def world_to_pixel (x, y, ox, oy, ori_ox, ori_oy, offset=(10+320+176, 192//2), size=320, angle_jitter=15):
        
    pixel_dx, pixel_dy = (x-ox)*PIXELS_PER_METER, (y-oy)*PIXELS_PER_METER
    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox
    
    pixel_x = -pixel_x
    # print('loc', pixel_x, pixel_y)
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
args = YamlConfig.from_nested_dicts(load_config('config/hound_straight.yaml'))
env = lmdb.open('/home/moonlab/Documents/karthik/dataset_384_160/{}'.format(sys.argv[1]))
pygame.init()
pygame.font.init()
display = pygame.display.set_mode(
                    (args.width, args.height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
display.fill((0,0,0))
font_name = 'courier' if os.name == 'nt' else 'mono'
fonts = [x for x in pygame.font.get_fonts() if font_name in x]
default_font = 'ubuntumono'
mono = default_font if default_font in fonts else fonts[0]
mono = pygame.font.match_font(mono)
_font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)

with env.begin() as txn:
    # measurement = np.frombuffer(txn.get(('rgb_left_%04d'%i).encode()), np.float32)

    length = int(txn.get(str('len').encode()))
    for i in range(length-25):
        rgb_left = np.fromstring(txn.get(('rgb_left_%04d'%i).encode()), np.uint8).reshape(160,384,3)
        rgb_right = np.fromstring(txn.get(('rgb_right_%04d'%i).encode()), np.uint8).reshape(160,384,3)
        bird_view = np.fromstring(txn.get(('birdview_%04d'%i).encode()), np.uint8).reshape(320,320,8)
        # removing traffic channels
        # bird_view = np.delete(bird_view, [2], axis=-1)
        measurement = np.frombuffer(txn.get(('measurements_%04d'%i).encode()), np.float32)
        display.blit(pygame.surfarray.make_surface(rgb_left.swapaxes(0, 1)), (0, 0))
        display.blit(pygame.surfarray.make_surface(rgb_right.swapaxes(0, 1)), (0, 160))
        display.blit(pygame.surfarray.make_surface(np.zeros((320, 320))), (704, 0))
        birdview = crop_birdview(bird_view)
        bird_view = visualize_birdview(bird_view)
        display.blit(pygame.surfarray.make_surface(np.transpose(bird_view, (1, 0, 2))), (384, 0))
        ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear, traffic_light  = measurement
        v_offset = 4
        bar_width = 106
        items = list()
        speed = 3.6 * math.sqrt(vx**2 + vy**2 + vz**2)
        speed_str = 'Speed:   % 15.3f km/h' % (3.6 * math.sqrt(vx**2 + vy**2 + vz**2))
        items.append(str(speed_str))

        # CONTROL

        items.append(('Throttle:   % 12.3f' % (throttle), throttle, 1))
        items.append(('Steer:   % 15.3f' % (steer), steer, -1))
        items.append(('Brake:   % 15.3f' % (brake), brake, 1))

        # COMMAND
        commands = {
                -1:"VOID",
                1: "LEFT",
                2:"RIGHT",
                3:"STRAIGHT",
                4:"LANEFOLLOW",
                5:"CHANGELANELEFT",
                6:"CHANGELANERIGHT"
        }
        items.append('Command:   % 15s' % (commands[cmd]))

        # TRAFFIC LIGHT

        items.append('Traffic Light:   % 12s' % ('RED' if traffic_light == 1.0 else 'GREEN'))
        
        #birdview
        # birdview = np.reshape(birdview, (192, 192, 5))
        # birdview = birdview_transform(birdview)
        # birdview = birdview[None, :].to(config['device'])
        # rgb_left = birdview_transform(rgb_left)
        # rgb_right = birdview_transform(rgb_right)
        # rgb_left = rgb_left[None, :].to(config['device'])
        # rgb_right = rgb_right[None, :].to(config['device'])
        # command = one_hot(torch.Tensor([cmd])).to(config['device'])
        # speed = torch.Tensor([float(speed)]).to(config['device'])
        # traffic = torch.Tensor([traffic_light]).to(config['device'])
        # print(command)
        # with torch.no_grad():
        #     # _image_location = image_net(rgb_left, rgb_right, speed, command, traffic)
        #     _teac_location = teacher_net(birdview, speed, command, traffic)
        # # _image_location = _image_location.squeeze().detach().cpu().numpy()
        # # _image_location = (_image_location +1) * np.array([800, 600])/2
        # coord_converter = CoordConverter()
        # teac_location = coord_converter(_teac_location)* np.array([800, 600])/2
        # # print(teac_location.shape)
        # _teac_location = (_teac_location + 1) * (0.5 * 192)/PIXELS_PER_METER
        # # print(_teac_location)
        # # print(_image_location)
        # for teac_loc in _teac_location[0]:
        #     pygame.draw.rect(display, RED, pygame.Rect(teac_loc[0]*5+320//2+1600, 192+teac_loc[1], 3, 3))
        # # for teac_loc in _image_location:
        # #     pygame.draw.rect(display, RED, pygame.Rect(teac_loc[0], teac_loc[1], 3, 3))
        # for teac_loc in teac_location[0]:
        #     pygame.draw.rect(display, RED, pygame.Rect(teac_loc[1], teac_loc[0], 3, 3))
        # print(_teac_location, teac_location)
        # print(coord_converter)
        # DISPLAY RENDERING
        for item in items:
            if isinstance(item, tuple):
                if item[-1] < 0:
                    surface = _font_mono.render(item[0], True, (255, 255, 255))
                    display.blit(surface, (1608, 320+v_offset))
                    rect_border = pygame.Rect((1800, 320+v_offset + 4), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    rect = pygame.Rect((1800 + (item[1]+1)/2 * (bar_width - 6), 320+v_offset + 4), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                    v_offset +=18
                else:    
                    surface = _font_mono.render(item[0], True, (255, 255, 255))
                    display.blit(surface, (1608, 320+v_offset))
                    v_offset +=18
            else:
                surface = _font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (1608, 320+v_offset))
                v_offset +=18
            # gap = 5
            # n_step = 5
            # ox, oy, oz, ori_ox, ori_oy  = measurement[:5]
            # gt_loc = []
            # # for dt in range(gap+15, gap*(n_step+1), gap):
            # dt = 0
            # while dt < 25 :
            #     index = i + dt
            #     f_measurement = np.frombuffer(txn.get(("measurements_%04d"%index).encode()), np.float32)
            #     x, y, z, ori_x, ori_y = f_measurement[:5]
            #     xx, yy = world_to_pixel(x, y, ox, oy, ori_ox, ori_oy, offset=(0, 0))/PIXELS_PER_METER
            #     thres_x, thres_y = xx, yy
            #     # if abs(xx) < 3.5:
            #     #     if abs(yy) > 2.0:
            #     #         gt_loc.append([-3.5, 2.0])
            #     #     else:
            #     #         gt_loc.append([-3.5, 0.0])
            #     # elif abs(xx) > 3.5:
            #     #     if abs(yy) > 2.0:
            #     #         gt_loc.append([xx, 2.0])
            #     #     else:
            #     #         gt_loc.append([xx, yy])
            #     # else:
            #     gt_loc.append([xx-3.5, yy])
            #     dt +=5
            # # print('gt', np.array(gt_loc)*5)
            # for loc in gt_loc:
            #     pixel_y = loc[0]*5
            #     pixel_x = loc[1]*5
            #     pygame.draw.rect(display, BLUE, pygame.Rect(pixel_x+320//2+1600, 260+pixel_y-14, 3, 3))
            # ROTATION_MATRIX = np.array([
            #     [1, 0, 0, 0],
            #     [0, 1, 0, 0],
            #     [0, 0, -1, 0],
            # ])

            # f = 384 /(2 * np.tan(90 * np.pi / 360))
            # #print(f)
            # A = np.array([
            #     [f, 0., 384/2],
            #     [0, f, 160/2],
            #     [0., 0., 1.]
            # ])
            # EXTRINSIC_ROTATION_LEFT = np.array([[0, 1, 0, -0.15],
            #                             [0, 0, 1, 0.88],
            #                             [1, 0, 0, 2.2]])
            # EXTRINSIC_ROTATION_RIGHT = np.array([[0, 1, 0, 0.5],
            #                             [0, 0, 1, -1.4],
            #                             [1, 0, 0, -2.0]])
            # for loc in gt_loc:
            #     # print(loc)
            #     point = np.array([loc[1]-0.15, +0.88, loc[0]+2.2, 1])
            #     #print('gt', point)
            #     point_left = np.matmul(np.matmul(A, ROTATION_MATRIX), point)
            #     point_left /= point_left[-1]
            #     point_right = np.matmul(np.matmul(A, ROTATION_MATRIX), np.concatenate([np.matmul(EXTRINSIC_ROTATION_RIGHT,point), [1]]))
            #     point_right /= point_right[-1]
            #     # point_left[0] = np.clip(point_left[0], 0, 770)
            #     # point_left[1] = np.clip(point_left[1], 0, 570)
            #     #print('point_left', point_left)
            #     if 0 < point_left[0] and point_left[0] < 800 and 0 < point_left[1] and  point_left[1] < 600:
            #         pygame.draw.rect(display, BLUE, pygame.Rect(point_left[0], point_left[1], 3, 3))
                #else: 
                    #print('XX')


            pygame.display.update()
    pygame.quit()
