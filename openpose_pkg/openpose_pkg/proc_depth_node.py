#!/usr/bin/env python

## we need the frequency
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
##from std_msgs.msg import Int16
import numpy as np
from openpose_interfaces.msg import *
sys.path.append('/home/mapir/ros2_ws/src/openpose_pkg/openpose_pkg')
from proc_depth import *
#import rosbag
from rclpy.exceptions import ROSInterruptException

import time
import rospkg #for the ros pkg path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading

class HumanDepthProcessorClass(Node):
    def __init__(self):
        super().__init__('proc_human_depth_2d_node')

        ###############################
        ####    PARAMETERS         ####
        ###############################
        ## debug flag. show info
        self.declare_parameter('~DebugInfo/debug_info', True)
        self._debug = self.get_parameter('~DebugInfo/debug_info').get_parameter_value().bool_value

        self.declare_parameter('~ImageParameters/factor', 0.2)
        self._factor = self.get_parameter('~ImageParameters/factor').get_parameter_value().double_value

        self.declare_parameter('~HumanDetected/compute_body', False)
        self._create_body_3d = self.get_parameter('~HumanDetected/compute_body').get_parameter_value().bool_value

        ## topics names
        self.declare_parameter('~ROSTopics/humans_3d_topic', '/users') ##sub
        self.declare_parameter('~ROSTopics/users_3d_topic', '/users3D') ##pub
        self.declare_parameter('~ROSTopics/marker_users_topic', '/markers_users') # array de markers en el espacio, cada uno marca una pose
        self.declare_parameter('~ROSServices/no_human_srv', '/human')
        self.declare_parameter('~ROSTopics/markers_3d_body_parts_topic', '/markers_body') # array de markers en el espacio, indicando las partes del humano en el espacio (lineas y esferas)

        self._topic_human_3d_name = self.get_parameter('~ROSTopics/humans_3d_topic').get_parameter_value().string_value ##sub
        self._topic_users_3d = self.get_parameter('~ROSTopics/users_3d_topic').get_parameter_value().string_value ##pub
        self._topic_marker = self.get_parameter('~ROSTopics/marker_users_topic').get_parameter_value().string_value
        self._no_human_srv = self.get_parameter('~ROSServices/no_human_srv').get_parameter_value().string_value
        self._topic_body_3d_markers = self.get_parameter('~ROSTopics/markers_3d_body_parts_topic').get_parameter_value().string_value

        #publishers and subscribers
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self._sub_humans = self.create_subscription(UserRGBDArray, self._topic_human_3d_name, self.callback_users, qos_profile) ## users from proc. image node. # array de UserRGBD (para cada usuario: nombre, coord. cara en img, pose, body parts...)
        self._sub_cloud = self.create_subscription(PointCloud2, '/cloud_topic', self.callback_cloud, qos_profile) 

        self._pub_user = self.create_publisher(User3DArray, self._topic_users_3d, qos_profile) ## array de User3D (para cada usuario: nombre, pose, body parts, altura...)
        self._pub_marker = self.create_publisher(MarkerArray, self._topic_marker, qos_profile) 
        self._pub_markers_body = self.create_publisher(MarkerArray, self._topic_body_3d_markers, qos_profile) ##img with drawn human


        self._header_users = None
        self._users = None
        self._width_img = 100.0
        self._height_img = 100.0
        self._cloud = PointCloud2()

        #####################
        ### DEBUG INFO    ###
        #####################
        if (self._debug):
            self.get_logger().info('Debug info activated')
            print('Topic name for the openpose-3d users: %s' % (self._topic_users_3d))
            print('Topic name for the users (subscriber): %s' % (self._topic_human_3d_name))

    def process_humans(self):
        thread = threading.Thread(target = rclpy.spin,args = (self,), daemon=True)
        thread.start()

        self._bridge = CvBridge()

        r = self.create_rate(500) 
        #self._k = 0
        while (rclpy.ok()):          # ANTES PONIA while(not rospy.is_shutdown()()
            r.sleep()
        
        rclpy._shutdown("Exiting openpose publisher node.\n")
        sys.exit(1)
    

    def callback_users(self, user_array):

        self.get_logger().info('CALLBACK_USERS')

        user_array.cloud = self._cloud
        user_array.compute_depth = 1
        self._users = user_array

        if len(self._users.users) > 0:
            #print('number of users: %d' % (len(self._users.users)))
            stamp = self.get_clock().now().to_msg()
            users = HumanDepthSet(self._users, self._factor, self._create_body_3d, stamp)
            userarray_msg, userarrarmarker_msg, markers_body, cloud_correct = users.create_msg_depth_cloud()
            ###print userarray_msg
            if cloud_correct:
                self.get_logger().info('cloud correct')
                print('number of users: %d' % (len(userarray_msg.users)))
                if len(userarray_msg.users) > 0:
                    self._pub_user.publish(userarray_msg)
                    self._pub_marker.publish(userarrarmarker_msg)
                    self.get_logger().info('publishing users3D')
                    if self._create_body_3d:
                        self._pub_markers_body.publish(markers_body)

    def callback_cloud(self, cloud):

        self._cloud = cloud
        self.get_logger().info('CALLBACK_CLOUD')

                        
def main(args=None):
    rclpy.init(args=args)

    try:
        x = HumanDepthProcessorClass()
        x.process_humans()
    except ROSInterruptException:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
