#!/usr/bin/env python

## we need the frequency
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
from openpose_interfaces.msg import *
sys.path.append('/home/mapir/ros2_ws/src/openpose_pkg/openpose_pkg')
from proc_image import *
#import rosbag
from rclpy.exceptions import ROSInterruptException

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading

class HumanProcessorClass(Node):
    def __init__(self):
        super().__init__('proc_human_2d_node')

        ###############################
        ####    PARAMETERS         ####
        ###############################
        ## debug flag. show info
        self.declare_parameter('DebugInfo.debug_info', True)
        self._debug = self.get_parameter('DebugInfo.debug_info').get_parameter_value().bool_value

        ## topics names
        self.declare_parameter('ROSTopics.humans_topic', '/humans') ##sub
        self.declare_parameter('ROSTopics.video_humans_drawn_topic', '/openpose/frame_humans/draw_img') ##pub
        self.declare_parameter('ROSTopics.humans_3d_topic', '/users') ##pub

        self._topic_humans_name = self.get_parameter('ROSTopics.humans_topic').get_parameter_value().string_value
        self._topic_human_frame_drawn_name = self.get_parameter('ROSTopics.video_humans_drawn_topic').get_parameter_value().string_value
        self._topic_human_3d_name = self.get_parameter('ROSTopics.humans_3d_topic').get_parameter_value().string_value

        self.declare_parameter('HumanDetected.threshold_human', 0.3)
        self._threshold = self.get_parameter('HumanDetected.threshold_human').get_parameter_value().double_value

        self._header_img = None

        self._humans = None
        self._frame_cv = None
        self._width_img = 100.0
        self._height_img = 100.0

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self._sub_humans = self.create_subscription(HumanArray2, self._topic_humans_name, self.callback_humans, qos_profile) ## humans from openpose
        
        self._pub_user = self.create_publisher(UserRGBDArray, self._topic_human_3d_name, qos_profile) ## first user array in """"3D"""""2
        self._pub_img_user = self.create_publisher(Image, self._topic_human_frame_drawn_name, qos_profile) ##img with drawn human


        #####################
        ### DEBUG INFO    ###
        #####################
        if (self._debug):
            self.get_logger().info('Debug info activated')
            print('Topic name for the openpose-humans: %s' % (self._topic_humans_name))
            print('Topic name for the users (publish): %s' % (self._topic_human_3d_name))

    
    def process_humans(self):
        thread = threading.Thread(target = rclpy.spin,args = (self,), daemon=True)
        thread.start()

        self._bridge = CvBridge()

        r = self.create_rate(500) 

        while (rclpy.ok()):
            r.sleep()

        rclpy.shutdown("Exiting openpose publisher node.\n")
        sys.exit(1)
    

    def callback_humans(self, human_array):
        self._header_img = human_array.header 
        image_depth = human_array.image_human

        try:
            frame_cv = self._bridge.imgmsg_to_cv2(image_depth.image_2d, "bgr8")
        except CvBridgeError as e:
            pass
                
        compute_depth = image_depth.valid_depth
        cloud = PointCloud2()
        if compute_depth:
            cloud = image_depth.point_cloud_3d

        image = None
        if len(human_array.humans) > 0:
            self._humans = human_array
              
            depth_humans2 = UserRGBDArray()
            humans = HumanImageSet(self._humans)
                       
            depth_humans2, image = humans.create_msg_user_image(self._threshold, frame = frame_cv, draw_image = True)
            depth_humans2.cloud = cloud
            depth_humans2.compute_depth = compute_depth

            if(len(depth_humans2.users) > 0):
                self._pub_user.publish(depth_humans2)

            else:
                pub_empty = UserRGBDArray()
                self._pub_user.publish(pub_empty)

        else:
            pub_empty = UserRGBDArray()
            self._pub_user.publish(pub_empty)



        if (image is None):  
            image = cv2.copyMakeBorder(frame_cv,0,0,0,0,cv2.BORDER_REPLICATE)

        image_msg = Image()
        image_msg.header = self._header_img
        try:
            image_msg = self._bridge.cv2_to_imgmsg(image, "bgr8")

            self._pub_img_user.publish(image_msg)  
        except CvBridgeError as e:
            pass                                    


def main(args=None):
    rclpy.init(args=args)

    try:
        x = HumanProcessorClass()
        x.process_humans()
    except ROSInterruptException: 
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()