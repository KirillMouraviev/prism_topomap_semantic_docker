#!/usr/bin/env python 

import rospy
from sensor_msgs.msg import Image 
from sensor_msgs.msg import CameraInfo

rospy.init_node("update_frame_id")

#Updating frame id for the error depth_front frame id does not match rgb_front frame id
class update_frame_id:
    def __init__(self):
        #subscribe to your specific sensors
        self.sub_raw = rospy.Subscriber("/habitat/rgb/image", Image, self.callback_raw)
        self.sub_depth = rospy.Subscriber("/habitat/depth/image", Image, self.callback_depth)
        self.sub_info = rospy.Subscriber("/habitat/rgb/camera_info", CameraInfo, self.callback_info)
        self.pub_raw = rospy.Publisher("/rgb/image_rect_color", Image, queue_size = 1)
        self.pub_depth = rospy.Publisher("/depth_registered/image_rect", Image, queue_size = 1)
        self.pub_info = rospy.Publisher("/rgb/camera_info", CameraInfo, queue_size = 1)
        self.camera_frame_id = rospy.get_param('~camera_frame_id', 'camera_link_noised')
    def callback_raw(self, message):
        message.header.frame_id = "camera_link_noised"
        self.pub_raw.publish(message)
    def callback_depth(self, message):
        message.header.frame_id = "camera_link_noised"
        self.pub_depth.publish(message)
    def callback_info(self, message):
        message.header.frame_id = "camera_link_noised"
        self.pub_info.publish(message)

update_frame_id = update_frame_id()
rospy.spin()

print("\nNode shutdown\n")
