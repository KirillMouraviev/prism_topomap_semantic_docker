#! /usr/bin/env python

import rospy
import numpy as np
import tf
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry


class PoseNoiser():
    def __init__(self):
        rospy.init_node('pose_noiser')
        np.random.seed(0)
        self.x_std = rospy.get_param('~x_std', 0.0075)
        self.y_std = rospy.get_param('~y_std', 0.00175)
        self.theta_std = rospy.get_param('~theta_std', 0.0)
        self.noise_update_interval = rospy.get_param('~noise_update_interval', 10.0)
        self.publish_odom = rospy.get_param('~publish_odom', True)
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.prev_stamp = None
        self.x_noised = None
        self.y_noised = None
        self.theta_noised = None
        self.odom_z = 0
        self.vx_noise = 0
        self.vy_noise = 0
        self.w_noise = 0
        self.noise_update_time = 0
        self.noised_pose_publisher = rospy.Publisher('pose_noised', PoseStamped, latch=True, queue_size=100)
        self.tfbr = tf.TransformBroadcaster()
        self.transform_publisher = rospy.Publisher('habitat/transform_noised', TransformStamped, latch=True, queue_size=100)
        if self.publish_odom:
            self.odom_publisher = rospy.Publisher('odom_noised', Odometry, latch=True, queue_size=100)
        self.pose_subscriber = rospy.Subscriber('true_pose', PoseStamped, self.pose_callback)


    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.odom_z = msg.pose.position.z
        orientation = msg.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        msg_time = msg.header.stamp.to_sec()
        if msg_time - self.noise_update_time > self.noise_update_interval:
            self.vx_noise += np.random.normal(0, self.x_std)
            self.vy_noise += np.random.normal(0, self.y_std)
            self.w_noise += np.random.normal(0, self.theta_std)
            self.noise_update_time = msg_time
        #print('Noise x, y, theta:', self.vx_noise, self.vy_noise, self.w_noise)
        vx = 0
        vy = 0
        w = 0
        if self.prev_x is not None:
            dt = msg_time - self.prev_stamp
            #print('Theta:', theta)
            #print('dx, dy:', x - self.prev_x, y - self.prev_y)
            vx = ((x - self.prev_x) * np.cos(self.prev_theta) + (y - self.prev_y) * np.sin(self.prev_theta)) / dt
            vy = (-(x - self.prev_x) * np.sin(self.prev_theta) + (y - self.prev_y) * np.cos(self.prev_theta)) / dt
            w = (theta - self.prev_theta) / dt
            if vx != 0 or vy != 0:
                vx += self.vx_noise
                vy += self.vy_noise
            #w += self.w_noised

            dx_noised = dt * (vx * np.cos(-self.theta_noised) + vy * np.sin(-self.theta_noised))
            dy_noised = dt * (-vx * np.sin(-self.theta_noised) + vy * np.cos(-self.theta_noised))
            #print('vx, vy:', vx, vy)
            #print('dx, dx noised:', x - self.prev_x, dx_noised)
            #print('dy, dy noised:', y - self.prev_y, dy_noised)
            dtheta_noised = dt * w
            self.x_noised += dx_noised
            self.y_noised += dy_noised
            self.theta_noised = theta + self.w_noise
            #print('GT x, y:', x, y)
            #print('Noised x, y:', self.x_noised, self.y_noised)
        else:
            self.x_noised = x
            self.y_noised = y
            self.theta_noised = theta

        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta
        self.prev_stamp = msg_time

        pose_noised_msg = PoseStamped()
        pose_noised_msg.header = msg.header
        pose_noised_msg.header.frame_id = 'odom'
        pose_noised_msg.pose = msg.pose
        pose_noised_msg.pose.position.x = self.x_noised
        pose_noised_msg.pose.position.y = self.y_noised
        pose_noised_msg.pose.orientation.x, \
        pose_noised_msg.pose.orientation.y, \
        pose_noised_msg.pose.orientation.z, \
        pose_noised_msg.pose.orientation.w = tf.transformations.quaternion_from_euler(0, 0, self.theta_noised)
        self.noised_pose_publisher.publish(pose_noised_msg)

        cur_transform = TransformStamped()
        cur_transform.header = pose_noised_msg.header
        cur_transform.child_frame_id = 'base_link_noised'
        cur_transform.transform.translation = pose_noised_msg.pose.position
        cur_transform.transform.rotation = pose_noised_msg.pose.orientation
        self.transform_publisher.publish(cur_transform)

        if self.publish_odom:
            odom_msg = Odometry()
            odom_msg.header = pose_noised_msg.header
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link_noised'
            odom_msg.pose.pose = pose_noised_msg.pose
            odom_msg.pose.covariance = list(np.eye(6).ravel() * 0.1)
            odom_msg.twist.twist.linear.x = vx
            odom_msg.twist.twist.linear.y = vy
            odom_msg.twist.twist.linear.z = 0
            odom_msg.twist.twist.angular.x = 0
            odom_msg.twist.twist.angular.y = 0
            odom_msg.twist.twist.angular.z = w
            odom_msg.twist.covariance = list(np.eye(6).ravel() * 0.01)
            self.odom_publisher.publish(odom_msg)


        self.tfbr.sendTransform((self.x_noised, self.y_noised, self.odom_z),
                           tf.transformations.quaternion_from_euler(0, 0, self.theta_noised),
                           msg.header.stamp,
                           'base_link_noised', 'odom')


    def run(self):
        rospy.spin()


pose_noiser = PoseNoiser()
pose_noiser.run()
