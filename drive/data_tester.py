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
from utils.train_utils import one_hot
BLUE = Color('blue')
PIXELS_PER_METER = 5
N_STEPS=5
CROP_SIZE = 192
MAP_SIZE=320
config = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'teacher_args' : {
                'model_path': '/workspace/LearningByCheating/training/birdview_new/model-861.th',
                }
            }
class CoordConverter():
    def __init__(self, w=800, h=600, fov=90, world_y=0.88, fixed_offset=3.5, device='cuda'):
        self._w = w
        self._h = h
        self._img_size = torch.FloatTensor([w,h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        
        self._tran = np.array([0.,0.,0.0])
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
    
        image_xy, _ = cv2.projectPoints(xyz, self._tran, self._rot, self._A, None)
        image_xy[...,0] = np.clip(image_xy[...,0], 0, self._w)
        image_xy[...,1] = np.clip(image_xy[...,1], 0, self._h)
    
        return image_xy[:,0]
    
    def __call__(self, map_locations):
        teacher_locations = map_locations.detach().cpu().numpy()
        teacher_locations = (teacher_locations + 1) * CROP_SIZE / 2
        N = teacher_locations.shape[0]
        teacher_locations[:,:,1] = CROP_SIZE - teacher_locations[:,:,1]
        teacher_locations[:,:,0] -= CROP_SIZE/2
        teacher_locations = teacher_locations / PIXELS_PER_METER
        teacher_locations[:,:,1] += self._fixed_offset
        teacher_locations = self._project_image_xy(np.reshape(teacher_locations, (N*N_STEP, 2)))
        teacher_locations = np.reshape(teacher_locations, (N,N_STEP,2))
        teacher_locations = torch.FloatTensor(teacher_locations)
        print(teacher_locations.shape)
        return teacher_locations

def world_to_pixel (x, y, ox, oy, ori_ox, ori_oy, offset=(10+320+176, 192//2), size=320, angle_jitter=15):
        
    pixel_dx, pixel_dy = (x-ox)*PIXELS_PER_METER, (y-oy)*PIXELS_PER_METER
    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox
    
    pixel_x = -pixel_x
    # print('loc', pixel_x, pixel_y)
    return np.array([pixel_x, pixel_y]) + offset
def crop_birdview(birdview, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview

args = YamlConfig.from_nested_dicts(load_config('config/hound_straight.yaml'))
for i in range(1):
    print('%03d' % i)
    env = lmdb.open('/workspace/dataset/train/'+('%03d' % i))

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
            rgb_left = np.fromstring(txn.get(('rgb_left_%04d'%i).encode()), np.uint8).reshape(600,800,3)
            rgb_right = np.fromstring(txn.get(('rgb_right_%04d'%i).encode()), np.uint8).reshape(600,800,3)
            bird_view = np.fromstring(txn.get(('birdview_%04d'%i).encode()), np.uint8).reshape(320,320,8)
            # removing traffic channels
            bird_view = np.delete(bird_view, [3, 4, 5], axis=-1)
            measurement = np.frombuffer(txn.get(('measurements_%04d'%i).encode()), np.float32)
            display.blit(pygame.surfarray.make_surface(rgb_left.swapaxes(0, 1)), (0, 0))
            display.blit(pygame.surfarray.make_surface(rgb_right.swapaxes(0, 1)), (800, 0))
            display.blit(pygame.surfarray.make_surface(np.zeros((320, 280))), (1600, 320))
            bird_view = crop_birdview(visualize_birdview(bird_view))
            display.blit(pygame.surfarray.make_surface(np.transpose(bird_view, (1, 0, 2))), (1600, 0))
            ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear, traffic_light  = measurement
            v_offset = 4
            bar_width = 106
            items = list()
            speed = 'Speed:   % 15.3f km/h' % (3.6 * math.sqrt(vx**2 + vy**2 + vz**2))
            items.append(str(speed))

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

            # Birdview model
            teacher_net = BirdViewPolicyModelSS(backbone='resnet18').to(config['device'])
            teacher_net.load_state_dict(torch.load(config['teacher_args']['model_path']))
            teacher_net.eval()
            birdview = bird_view
            birdview = birdview.to(config['device'])
            command = one_hot(command).to(config['device'])
            speed = speed.to(config['device'])
            traffic = traffic.to(config['device'])
            with torch.no_grad():
                _teac_location = teacher_net(birdview, speed, command, traffic)
            print(_teach_location)
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
                        rect_border = pygame.Rect((1800, 320+v_offset + 4), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        rect = pygame.Rect((1800, 320+v_offset+4), (item[1] * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                        v_offset +=18
                else:
                    surface = _font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (1608, 320+v_offset))
                    v_offset +=18
            gap = 5
            n_step = 5
            ox, oy, oz, ori_ox, ori_oy  = measurement[:5]
            gt_loc = []
            # for dt in range(gap+15, gap*(n_step+1), gap):
            dt = 0
            while dt <= 25 :
                index = i + dt
                f_measurement = np.frombuffer(txn.get(("measurements_%04d"%index).encode()), np.float32)
                x, y, z, ori_x, ori_y = f_measurement[:5]
                xx, yy = world_to_pixel(x, y, ox, oy, ori_ox, ori_oy, offset=(0, 0))/PIXELS_PER_METER
                thres_x, thres_y = xx, yy
                # if abs(xx) < 6:
                #     gt_loc.append([-6, 0])
                # else:
                gt_loc.append([xx, yy])
                dt +=5
            for loc in gt_loc:
                pixel_y = loc[0]*5
                pixel_x = loc[1]*5
                pygame.draw.rect(display, BLUE, pygame.Rect(pixel_x+190//2+1600, 192+pixel_y-10, 3, 3))
            ROTATION_MATRIX = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
            ])

            f = 800 /(2 * np.tan(90 * np.pi / 360))
            #print(f)
            A = np.array([
                [f, 0., 800/2],
                [0, f, 600/2],
                [0., 0., 1.]
            ])
            EXTRINSIC_ROTATION_LEFT = np.array([[0, 1, 0, -0.15],
                                        [0, 0, 1, 0.88],
                                        [1, 0, 0, 2.2]])
            EXTRINSIC_ROTATION_RIGHT = np.array([[0, 1, 0, 0.5],
                                        [0, 0, 1, -1.4],
                                        [1, 0, 0, -2.0]])
            for loc in gt_loc:
                print(loc)
                point = np.array([loc[1]-0.15, +0.88, loc[0]+2.2, 1])
                #print('gt', point)
                point_left = np.matmul(np.matmul(A, ROTATION_MATRIX), point)
                point_left /= point_left[-1]
                point_right = np.matmul(np.matmul(A, ROTATION_MATRIX), np.concatenate([np.matmul(EXTRINSIC_ROTATION_RIGHT,point), [1]]))
                point_right /= point_right[-1]
                #print('point_left', point_left)
                if 0 < point_left[0] and point_left[0] < 800 and 0 < point_left[1] and  point_left[1] < 600:
                    pygame.draw.rect(display, BLUE, pygame.Rect(point_left[0], point_left[1], 3, 3))
                #else: 
                    #print('XX')


            pygame.display.update()
    pygame.quit()
