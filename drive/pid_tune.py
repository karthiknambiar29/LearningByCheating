import numpy as np
import lmdb
import cv2
import pygame
import torch
import sys
import glob
from data_util import YamlConfig, load_config
from pygame.locals import Color

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
    sys.path.append(glob.glob('../drive')[0])
    sys.path.append('../LearningByCheating')
except IndexError as e:
    pass
from models.image import ImagePolicyModelSS
from utils.train_utils import one_hot
from torchvision import transforms
birdview_transform = transforms.ToTensor()
import pandas as pd
BLUE = Color('blue')
RED = Color('red')
ORANGE = Color('orange')
PIXELS_PER_METER = 5
N_STEP=5
CROP_SIZE = 320
MAP_SIZE=320
config = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'image_args' : {
                'model_path': '/home/moonlab/Documents/LearningByCheating/training/image_direct_unbiased_traffic_iitj/model-915.th',
                }
            }
image_net = ImagePolicyModelSS(backbone='resnet34').to(config['device'])
image_net.load_state_dict(torch.load(config['image_args']['model_path']))
image_net.eval()
from models.controller import CustomController, PIDController
from models.controller import ls_circle
from models import common
from sklearn.metrics import mean_squared_error

class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=0.88, fixed_offset=0.0, device='cuda'):
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
import os
args = YamlConfig.from_nested_dicts(load_config('config/hound_config.yaml'))
# env = lmdb.open('/home/moonlab/Documents/LearningByCheating/dataset/train/{}'.format(sys.argv[1]))
pygame.init()
pygame.font.init()
display = pygame.display.set_mode(
                    (args.width, args.height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
display.fill((0,0,0))


def unproject(output, world_y=0.88, fov=90):

    cx, cy = np.array([384, 160]) / 2
    
    w, h = np.array([384, 160])
    
    f = w /(2 * np.tan(fov * np.pi / 360))
    
    xt = (output[...,0:1] - cx) / f
    yt = (output[...,1:2] - cy) / f
    
    world_z = world_y / yt
    world_x = world_z * xt
    
    world_output = np.stack([world_x, world_z],axis=-1)
    
    world_output = world_output.squeeze()
    
    return world_output

steer_points = {"1": 4, "2": 4, "3": 2, "4": 4}
df = pd.DataFrame(columns=['S_KP', 'S_KI', 'S_KD', 'T_KP', 'T_KI', 'T_KD', 'loss_T', 'loss_S'])
import itertools

hyper_params = {
    'S_KP': np.arange(0, 1, 0.1),
    'S_KI': np.arange(0, 1, 0.1),
    'S_KD': np.arange(0, 1, 0.1),
    'T_KP': np.arange(0, 1, 0.1),
    'T_KI': np.arange(0, 1, 0.1),
    'T_KD': np.arange(0, 1, 0.1),
}

a = hyper_params.values()
combinations = list(itertools.product(*a))
for c in combinations[:2]:
    print(c)
    pid = {
    "1" : {"Kp": c[0], "Ki": c[1], "Kd":c[2]}, # Left
    "2" : {"Kp": c[0], "Ki": c[1], "Kd":c[2]}, # Right
    "3" : {"Kp": c[0], "Ki": c[1], "Kd":c[2]}, # Straight
    "4" : {"Kp": c[0], "Ki": c[1], "Kd":c[2]}, # Follow
    }
    turn_control = CustomController(pid, dt=0.05)
    speed_control = PIDController(K_P=c[3], K_I=c[4], K_D=c[5], fps=20)
    pred_t = []
    pred_s = []
    pred_b = []
    gt_t = []
    gt_b = []
    gt_s = []
    CMD = []
    env = lmdb.open('/home/moonlab/Documents/LearningByCheating/dataset/train/001')
    with env.begin() as txn:
        length = int(txn.get(str('len').encode()))
        for i in range(length-25):
            rgb_left = np.fromstring(txn.get(('rgb_left_%04d'%i).encode()), np.uint8).reshape(160,384,3)
            rgb_right = np.fromstring(txn.get(('rgb_right_%04d'%i).encode()), np.uint8).reshape(160,384,3)
            bird_view = np.fromstring(txn.get(('birdview_%04d'%i).encode()), np.uint8).reshape(320,320,8)
            # removing traffic channels
            measurement = np.frombuffer(txn.get(('measurements_%04d'%i).encode()), np.float32)
            display.blit(pygame.surfarray.make_surface(rgb_left.swapaxes(0, 1)), (0, 0))
            display.blit(pygame.surfarray.make_surface(rgb_right.swapaxes(0, 1)), (0, 160))
            display.blit(pygame.surfarray.make_surface(np.zeros((320, 320))), (704, 0))
            ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, gt_steer, gt_throttle, gt_brake, manual, gear, traffic_light  = measurement

            rgb_left = birdview_transform(rgb_left)
            rgb_right = birdview_transform(rgb_right)
            rgb_left = rgb_left[None, :].to(config['device'])
            rgb_right = rgb_right[None, :].to(config['device'])
            traffic_light = torch.Tensor([traffic_light]).to(config['device'])
            command = one_hot(torch.Tensor([cmd])).to(config['device'])
            speed = np.sqrt(vx**2 + vy**2+vz**2)
            # print(float(speed)*18/5)
            speed = torch.Tensor([float(speed)]).to(config['device'])
            with torch.no_grad():
                _image_locations = image_net(rgb_left, rgb_right, speed, command, traffic_light)
            
            _image_locations = _image_locations.squeeze().detach().cpu().numpy()
            _world_locations = unproject((_image_locations + 1)*np.array([384, 160])/2)
            for x, y in (_image_locations+1) * (0.5*np.array([384, 160])):
                pygame.draw.rect(display, RED, pygame.Rect(int(x), int(y), 3, 3))
            
            targets = [(0, 0)]

            for i in range(5):
                pixel_dx, pixel_dy = _world_locations[i]
                angle = np.arctan2(pixel_dx, pixel_dy)
                dist = np.linalg.norm([pixel_dx, pixel_dy])

                targets.append([dist * np.cos(angle), dist * np.sin(angle)])

            targets = np.array(targets)
            target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (5 * 0.05)

            c, r = ls_circle(targets)
            n = steer_points.get(str(int(cmd)), 1)
            closest = common.project_point_to_circle(targets[n], c, r)
            
            
            
            v = [1.0, 0.0, 0.0]
            w = [closest[0], closest[1], 0.0]
            alpha = common.signed_angle(v, w)
            acceleration = target_speed - speed
            steer = turn_control.run_step(alpha, int(cmd))
            throttle = speed_control.step(acceleration)
            brake = 0.0
            steer = np.clip(steer, -1.0, 1.0)
            throttle = np.clip(throttle.detach().cpu(), 0.0, 0.75)
            brake = np.clip(brake, 0.0, 1.0)

            pred_t.append(throttle)
            pred_s.append(steer)
            pred_b.append(brake)
            gt_t.append(gt_throttle)
            gt_s.append(gt_steer)
            gt_b.append(gt_brake)
            CMD.append(cmd)

            

            

            pygame.display.update()
        pred_t = np.array(pred_t)
        pred_s = np.array(pred_s)
        pred_b = np.array(pred_b)
        gt_t = np.array(gt_t)
        gt_s = np.array(gt_s)
        gt_b = np.array(gt_b)
        CMD = np.array(CMD)
        dict = {'S_KP': c[0], 'S_KI': c[1], 'S_KD': c[2], 'T_KP': c[3], 'T_KI': c[4], 'T_KD': c[5], 'loss_T':mean_squared_error(pred_t, gt_t), 'loss_S':mean_squared_error(pred_s, gt_s)}
        df = df.append(dict, ignore_index=True)
    # df.loc[i] = [c[0]] + [c[1]] +  [c[2]] +  [c[3]] +  [c[4]] +  [c[5]] + [mean_squared_error(pred_t, gt_t)] + [mean_squared_error(pred_s, gt_s)]
df.to('tune.csv')


    # tune = np.vstack((pred_t, pred_s, pred_b, gt_t, gt_s, gt_b, CMD))

pygame.quit()
