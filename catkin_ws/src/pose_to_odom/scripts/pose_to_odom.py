#! /usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np

positions = []
orientations = []
timestamps = []


def pose_callback(msg):
	odom_msg = Odometry()
	odom_msg.header = msg.header
	odom_msg.header.frame_id = 'map'
	odom_msg.child_frame_id = 'base_link'
	odom_msg.pose.pose = msg.pose
	odom_msg.pose.covariance = list(np.eye(6).ravel() * 0.1)
	odom_msg.twist.covariance = list(np.eye(6).ravel() * 0.1)
	cur_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
	cur_orientation = tf.transformations.euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
	cur_stamp = msg.header.stamp.to_sec()
	if len(positions) > 0:
		dt = cur_stamp - timestamps[-1]
		d_pos = np.array(cur_position) - np.array(positions[-1])
		angle = orientations[-1][2]
		pose_shift = [0, 0, d_pos[2]]
		pose_shift[0] = d_pos[0] * np.cos(angle) + d_pos[1] * np.sin(angle)
		pose_shift[1] = -d_pos[0] * np.sin(angle) + d_pos[1] * np.cos(angle)
		odom_msg.twist.twist.linear.x = pose_shift[0] / dt
		odom_msg.twist.twist.linear.y = pose_shift[1] / dt
		odom_msg.twist.twist.linear.z = pose_shift[2] / dt
		odom_msg.twist.twist.angular.x = (cur_orientation[0] - orientations[-1][0]) / dt
		odom_msg.twist.twist.angular.y = (cur_orientation[1] - orientations[-1][1]) / dt
		odom_msg.twist.twist.angular.z = (cur_orientation[2] - orientations[-1][2]) / dt
	positions.append(cur_position)
	orientations.append(cur_orientation)
	timestamps.append(cur_stamp)
	odom_publisher.publish(odom_msg)


rospy.init_node('pose_to_odom')
odom_publisher = rospy.Publisher('/odom', Odometry, latch=True, queue_size=100)
pose_subscriber = rospy.Subscriber('/true_pose', PoseStamped, pose_callback)
rospy.spin()