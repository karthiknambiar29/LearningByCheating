#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import TwistStamped
import numpy as np
import message_filters
from std_msgs.msg import Int32
import cv2
import sys
import glob
import torch
try:
    sys.path.append(glob.glob('/home/moonlab/Documents/LearningByCheating/bird_view')[0])
    sys.path.append('/home/moonlab/Documents/LearningByCheating')
except IndexError as e:
    pass
from models.image import ImagePolicyModelSS
from utils.train_utils import one_hot
from torchvision import transforms
transform = transforms.ToTensor()

class LBCros(object):
    def __init__(self):
        self.prev_vel = 0
        self.image_net = ImagePolicyModelSS(backbone='resnet34', all_branch=True).to('cuda')
        self.image_net.load_state_dict(torch.load('/home/moonlab/Documents/LearningByCheating/training/image_direct_unbiased_iitj/model-227.th'))
        self.image_net.eval()
        
        self.K = np.array([880.9860238607931, 0.0, 716.9382674369459, 0.0, 883.7308060193589, 492.4889256123, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.D = np.array([-0.3326947210797678, 0.08335145311385134, 0.00020372921734446254, -0.0017331122728458904, 0.0])
        rospy.init_node('listener', anonymous=True)
        
        self.pub0 = rospy.Publisher('/cam0_undist', Image, queue_size=10)
        self.pub1 = rospy.Publisher('/cam1_undist', Image, queue_size=10)
        self.cmd = 4

        cam0 = message_filters.Subscriber("/camera_array/cam0/image_raw", Image)
        cam1 = message_filters.Subscriber("/camera_array/cam1/image_raw", Image)
        rospy.Subscriber("/command", Int32, self.command)

        velocity = message_filters.Subscriber("/mavros/imu/data_raw", Imu)
        ts = message_filters.ApproximateTimeSynchronizer([cam0, cam1], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        

    def callback(self, cam0_msg, cam1_msg):
        cam0 = np.frombuffer(cam0_msg.data, dtype=np.uint8).reshape(cam0_msg.height, cam0_msg.width, -1)
        cam1 = np.frombuffer(cam1_msg.data, dtype=np.uint8).reshape(cam1_msg.height, cam1_msg.width, -1)
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
        speed = torch.Tensor([float(18*5/18)]).to('cuda')
        with torch.no_grad():
                _image_locations, preds = self.image_net(rgb_left, rgb_right, speed, command)
        _image_locations = _image_locations.squeeze().detach().cpu().numpy()

        for x, y in (_image_locations+1) * (0.5*np.array([384, 160])):
            cam0 = cv2.circle(cam0, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)
            cam1 = cv2.circle(cam1, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        cam0_msg.data = cam0.tobytes()
        cam0_msg.width = 384
        cam0_msg.height = 160
        cam1_msg.data = cam1.tobytes()
        cam1_msg.width = 384
        cam1_msg.height = 160

        self.pub0.publish(cam0_msg)
        self.pub1.publish(cam1_msg)

        

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():      
            r.sleep()

    def command(self, cmd):
        self.cmd = cmd

if __name__ == '__main__':
    xx = LBCros()
    xx.run()

