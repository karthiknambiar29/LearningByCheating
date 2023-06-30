#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped


class Command(object):
    def __init__(self):
        self.value = 0.0
        self.vels = []
        self.vel = 0.0
        rospy.init_node('echoer')

        self.pub = rospy.Publisher('/speed', Float32, queue_size=10, latch=True)
        rospy.Subscriber('/mavros/global_position/raw/gps_vel', TwistStamped, self.update_value)

    def update_value(self, msg):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z
        
        self.value = np.sqrt(vx**2+vy**2+vz**2)
        self.vels.append(self.value)

        # moving average
        if len(self.vels)>20:
            self.vel = np.mean(self.moving_average(self.vels, 3))
            
        else:
            self.vel = self.value
        if len(self.vels) > 100:
            self.vels = self.vels[-100:]
            print('del')
        

    def moving_average(self, x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.pub.publish(self.vel+5)
            r.sleep()


if __name__ == '__main__':
    xx = Command()
    xx.run()