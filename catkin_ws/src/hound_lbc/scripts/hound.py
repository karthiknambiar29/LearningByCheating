#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import TwistStamped
import numpy as np
import message_filters
from std_msgs.msg import Int32, Float32
import cv2
import sys
import glob
import torch
try:
    sys.path.append(glob.glob('/home/moonlab/Documents/LearningByCheating/bird_view')[0])
    sys.path.append('/home/moonlab/Documents/LearningByCheating')
except IndexError as e:
    pass
from models.controller import ls_circle
from models.image import ImagePolicyModelSS
from utils.train_utils import one_hot
from models.controller import CustomController, PIDController
from models import common
from torchvision import transforms
transform = transforms.ToTensor()

class LBCros(object):
    def __init__(self):
        self.img_size = np.array([384, 160])
        self.vel = 0.0
        self.image_net = ImagePolicyModelSS(backbone='resnet34', all_branch=True).to('cuda')
        self.image_net.load_state_dict(torch.load('/home/moonlab/Documents/LearningByCheating/training/image_direct_biased_iitj/model-757.th'))
        self.image_net.eval()
        
        self.K = np.array([880.9860238607931, 0.0, 716.9382674369459, 0.0, 883.7308060193589, 492.4889256123, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.D = np.array([-0.3326947210797678, 0.08335145311385134, 0.00020372921734446254, -0.0017331122728458904, 0.0])
        rospy.init_node('listener', anonymous=True)
        
        self.pub0 = rospy.Publisher('/cam0_undist', Image, queue_size=10)
        self.pub1 = rospy.Publisher('/cam1_undist', Image, queue_size=10)
        self.steer = rospy.Publisher('/steer', Float32, queue_size=10)
        self.throttle = rospy.Publisher('/throttle', Float32, queue_size=10)
        self.brake = rospy.Publisher('/brake', Float32, queue_size=10)

        self.cmd = 3

        cam0 = message_filters.Subscriber("/camera_array/cam0/image_raw", Image)
        cam1 = message_filters.Subscriber("/camera_array/cam1/image_raw", Image)
        rospy.Subscriber("/command", Int32, self.command)
        rospy.Subscriber("/speed", Float32, self.speed)
        self.steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}
        pid = {
                "1" : {"Kp": 1.0, "Ki": 0.00, "Kd":0.0}, # Left
                "2" : {"Kp": 1.0, "Ki": 0.00, "Kd":0.0}, # Right
                "3" : {"Kp": 1.0, "Ki": 0.00, "Kd":0.0}, # Straight
                "4" : {"Kp": 1.0, "Ki": 0.00, "Kd":0.0}, # Follow
            }
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P=1.0, K_I=.00, K_D=2.5)


        ts = message_filters.ApproximateTimeSynchronizer([cam0, cam1], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        

    def callback(self, cam0_msg, cam1_msg):
        cam1 = np.frombuffer(cam0_msg.data, dtype=np.uint8).reshape(cam0_msg.height, cam0_msg.width, -1)
        cam0 = np.frombuffer(cam1_msg.data, dtype=np.uint8).reshape(cam1_msg.height, cam1_msg.width, -1)
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (cam0.shape[1], cam0.shape[0]), 1, (cam0.shape[1], cam0.shape[0]))
        cam0 = cv2.undistort(cam0, self.K, self.D, None, newcameramatrix)
        cam1 = cv2.undistort(cam1, self.K, self.D, None, newcameramatrix)
        x,y,w,h = roi
        cam0 = cam0[y:y+h, x:x+w]
        cam1 = cam1[y:y+h, x:x+w]
        cam0 = cv2.resize(cam0, (384, 240))
        cam1 = cv2.resize(cam1, (384, 240))

        cam0 = cam0[40:-40, :]
        cam1 = cam1[40:-40, :]
       
        # Model
        cam0 = cv2.flip(cam0, 1)
        cam1 = cv2.flip(cam1, 1)
        rgb_left = transform(cam0)
        rgb_right = transform(cam1)
        rgb_left = rgb_left[None, :].to('cuda')
        rgb_right = rgb_right[None, :].to('cuda')
        command = one_hot(torch.Tensor([int(self.cmd.data)])).to('cuda')
        speed = torch.Tensor([float(self.vel)]).to('cuda')
        with torch.no_grad():
                _image_locations, preds = self.image_net(rgb_left, rgb_right, speed, command)
        _image_locations = _image_locations.squeeze().detach().cpu().numpy()
        _image_locations = (_image_locations+1)*(self.img_size/2)
        cam0 = cv2.putText(cam0, 'Left', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cam1 = cv2.putText(cam1, 'Right', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        for x, y in _image_locations:
            cam0 = cv2.circle(cam0, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)
            cam1 = cv2.circle(cam1, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)
        

        # controls
        steer, throttle, brake = self.control(_image_locations)

        cam0_msg.data = cam0.tobytes()
        cam0_msg.width = 384
        cam0_msg.height = 160
        cam1_msg.data = cam1.tobytes()
        cam1_msg.width = 384
        cam1_msg.height = 160

        self.pub0.publish(cam0_msg)
        self.pub1.publish(cam1_msg)
        self.steer.publish(steer)
        self.throttle.publish(throttle)
        self.brake.publish(brake)

        

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():      
            r.sleep()

    def command(self, cmd):
        self.cmd = cmd

    def speed(self, speed):
        self.vel = speed.data
    
    def unproject(self, output, world_y=0.88, fov=90):

        cx, cy = self.img_size / 2
        
        w, h = self.img_size
        
        f = w /(2 * np.tan(fov * np.pi / 360))
        
        xt = (output[...,0:1] - cx) / f
        yt = (output[...,1:2] - cy) / f
        
        world_z = world_y / yt
        world_x = world_z * xt
        
        world_output = np.stack([world_x, world_z],axis=-1)
        
        
        world_output = world_output.squeeze()
        
        return world_output*5
    
    def control(self, model_pred):
        engine_brake_threshold = 7.0
        brake_threshold = 7.0
        world_pred = self.unproject(model_pred)

        targets = [(0, 0)]

        for i in range(5):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)

        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (0.5)
        print(target_speed)
        c, r = ls_circle(targets)
        n = self.steer_points.get(str(self.cmd.data), 1)
        closest = common.project_point_to_circle(targets[n], c, r)
        
        acceleration = target_speed - self.vel
        
        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)

        steer = self.turn_control.run_step(alpha*2.75, self.cmd.data)
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        if target_speed <= engine_brake_threshold:
            steer = 0.0
            throttle = 0.0
            brake = 1.0
        if target_speed > 50/3:
            throttle = 0.0
            brake = 1.0


        steer, throttle, brake = self.postprocess(steer, throttle, brake)

        return steer, throttle, brake

    def postprocess(self, steer, throttle, brake):
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return steer, throttle, brake

if __name__ == '__main__':
    xx = LBCros()
    xx.run()

