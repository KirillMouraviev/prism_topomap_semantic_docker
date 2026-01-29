#!/usr/bin/env python 

import rospy
import numpy as np
np.float = np.float64
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from threading import Lock

rospy.init_node("merge_point_clouds")

#Updating frame id for the error depth_front frame id does not match rgb_front frame id
class PointCloudMerger:
    def __init__(self):
        #subscribe to your specific sensors
        self.sub_cloud1 = rospy.Subscriber('/habitat/points1', PointCloud2, self.callback1)
        self.sub_cloud2 = rospy.Subscriber('/habitat/points2', PointCloud2, self.callback2)
        self.sub_cloud3 = rospy.Subscriber('/habitat/points3', PointCloud2, self.callback3)
        self.sub_cloud4 = rospy.Subscriber('/habitat/points4', PointCloud2, self.callback4)
        self.pub_cloud = rospy.Publisher('/habitat/points', PointCloud2, queue_size=1)
        self.output_frame_id = rospy.get_param('~output_frame_id', 'camera_link')
        self.clouds2 = {}
        self.clouds3 = {}
        self.clouds4 = {}
        self.mutex = Lock()

    def find_sync_cloud(self, stamp, clouds):
        self.mutex.acquire()
        keys = list(clouds.keys())
        for ts in keys:
            if stamp.to_sec() - ts.to_sec() > 1.5:
                clouds.pop(ts)
            if abs(ts.to_sec() - stamp.to_sec()) < 0.03:
                self.mutex.release()
                return clouds[ts]
        self.mutex.release()
        return None

    def rotate(self, cloud, angle):
        cloud_rotated = cloud.copy()
        cloud_rotated['x'] = cloud['x'] * np.cos(angle) + cloud['z'] * np.sin(angle)
        cloud_rotated['y'] = cloud['y']
        cloud_rotated['z'] = -cloud['x'] * np.sin(angle) + cloud['z'] * np.cos(angle)
        cloud_rotated['r'] = cloud['r']
        cloud_rotated['g'] = cloud['g']
        cloud_rotated['b'] = cloud['b']
        return cloud_rotated

    def transform_to_base_scan(self, cloud):
        cloud_transformed = cloud.copy()
        cloud_transformed['x'] = cloud['z']
        cloud_transformed['y'] = -cloud['x']
        cloud_transformed['z'] = -cloud['y']
        return cloud_transformed

    def msg_to_array(self, msg):
        points_numpify = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        points_numpify = ros_numpy.point_cloud2.split_rgb_field(points_numpify)
        return points_numpify

    def callback1(self, msg):
        # wait for clouds 2 and 3 with the same stamp
        #print('Received cloud 1 at stamp {}'.format(msg.header.stamp.to_sec()))
        cloud1 = self.msg_to_array(msg)
        wait_start_time = rospy.Time.now().to_sec()
        while self.find_sync_cloud(msg.header.stamp, self.clouds2) is None or \
              self.find_sync_cloud(msg.header.stamp, self.clouds3) is None or \
              self.find_sync_cloud(msg.header.stamp, self.clouds4) is None:
            rospy.sleep(1e-4)
            if rospy.Time.now().to_sec() - wait_start_time > 0.2:
                return
        cloud2 = self.find_sync_cloud(msg.header.stamp, self.clouds2)
        cloud3 = self.find_sync_cloud(msg.header.stamp, self.clouds3)
        cloud4 = self.find_sync_cloud(msg.header.stamp, self.clouds4)
        cloud2_rotated = self.rotate(cloud2, -np.pi / 2)
        cloud3_rotated = self.rotate(cloud3, np.pi)
        cloud4_rotated = self.rotate(cloud4, np.pi / 2)
        cloud_merged = np.concatenate([cloud1, cloud2_rotated, cloud3_rotated, cloud4_rotated], axis=0)
        cloud_merged_transformed = self.transform_to_base_scan(cloud_merged)
        #print('Mean of the point cloud:', cloud_merged['x'].mean(), cloud_merged['y'].mean(), cloud_merged['z'].mean())
        cloud_merged_rgb = ros_numpy.point_cloud2.merge_rgb_fields(cloud_merged_transformed)
        cloud_merged_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_merged_rgb, stamp=msg.header.stamp, frame_id=self.output_frame_id)
        self.pub_cloud.publish(cloud_merged_msg)

    def callback2(self, msg):
        #print('Received cloud 2 at stamp {}'.format(msg.header.stamp.to_sec()))
        self.mutex.acquire()
        points = self.msg_to_array(msg)
        self.clouds2[msg.header.stamp] = points
        self.mutex.release()

    def callback3(self, msg):
        #print('Received cloud 3 at stamp {}'.format(msg.header.stamp.to_sec()))
        self.mutex.acquire()
        points = self.msg_to_array(msg)
        self.clouds3[msg.header.stamp] = points
        self.mutex.release()

    def callback4(self, msg):
        #print('Received cloud 4 at stamp {}'.format(msg.header.stamp.to_sec()))
        self.mutex.acquire()
        points = self.msg_to_array(msg)
        self.clouds4[msg.header.stamp] = points
        self.mutex.release()

    def run(self):
        rospy.spin()

pcd_merger = PointCloudMerger()
pcd_merger.run()

print("\nNode shutdown\n")
