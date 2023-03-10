#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from openpose_interfaces.msg import *
from openpose_interfaces.srv import *

#import os
import time
import numpy as np
import sys
#import rospkg #for the ros pkg path
### Load model ###
sys.path.append('/home/mapirs/ros2_openpose/src/openpose_pkg/openpose_pkg')
from estimator2 import TfPoseEstimator
#opencv to copy the image
import cv2
#import rosbag
from rclpy.exceptions import ROSInterruptException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading

#########################################################
##### OPENPOSE detects humans when an image arrived #####
#########################################################
## ToDo: improve detection
        
class OpenposeClass(Node):
    def __init__(self):
        super().__init__('openpose_node')

        self._img_ready = False
        self._detect_humans = False
        self._tf_start = False
        self._low_frec = True
        self._off_frec = False
        self._vel_window = [0.0, 0.0] ## LATER 
        self._window_param = [0.0, 0.0, 100.0, 100.0] ## LATER
        self._compute_depth = False
        self._move_robot = False

        #rospack = rospkg.RosPack()      
        self._models_path = '/home/mapirs/ros2_openpose/src/openpose_pkg/models/graph_opt.pb'

        #for opencv
        self._bridge = CvBridge()

        ###############################
        ####    PARAMETERS         ####
        ###############################
        ## debug flag. show info
        self.declare_parameter('~DebugInfo/debug_info', True)
        self._debug = self.get_parameter('~DebugInfo/debug_info').get_parameter_value().bool_value

        # how many humans are we gonna detect ??
        self.declare_parameter('~HumanDetected/single_human', False)
        self._single_human = self.get_parameter('~HumanDetected/single_human').get_parameter_value().bool_value

        ## threshold for the confidence of the detection
        self.declare_parameter('~HumanDetected/threshold_human', 0.0)
        self._threshold_human = self.get_parameter('~HumanDetected/threshold_human').get_parameter_value().double_value

        ## topics names
        self.declare_parameter('~ROSTopics/image_topic', 'openpose/usb_cam/image_dim_CODE')
        self.declare_parameter('~ROSTopics/humans_topic', '/humans')
        #self.declare_parameter('~ROSTopics/frame_humans_topic', '/frame_humans')
        #self.declare_parameter('~ROSServices/change_cam', '/off_frec')
        self.declare_parameter('~ROSServices/no_human_srv', '/human')

        self._topic_image_name =  self.get_parameter('~ROSTopics/image_topic').get_parameter_value().string_value
        self._topic_humans_name = self.get_parameter('~ROSTopics/humans_topic').get_parameter_value().string_value
        ## self._topic_human_frame_name = self.get_parameter('~ROSTopics/frame_humans_topic').get_parameter_value().string_value
        #self._srv_change_cam = self.get_parameter('~ROSServices/change_cam').get_parameter_value().string_value
        self._no_human_srv = self.get_parameter('~ROSServices/no_human_srv').get_parameter_value().string_value

        #self._srv_change_cam_handle = self.ServiceProxy(self._srv_change_cam, ChangeCam)         # el create service es cuando voy a definirlo con callback

        #publishers and subscribers
	#qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        ## cam topic
        self._sub_cam = self.create_subscription(ImageDepthHuman, self._topic_image_name, self.callback_image, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
        ## humans parts and image where they were found topics
        self._pub_human = self.create_publisher(HumanArray2, self._topic_humans_name, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))

        self._openpose = TfPoseEstimator(self._models_path)

        self._k = 0
        ###self._bag = rosbag.Bag('humans_openpose.bag', 'w')
        ###self._bag_time = rosbag.Bag('times_humans_openpose.bag', 'w')

        #####################
        ### DEBUG INFO    ###
        #####################
        if (self._debug):
            self.get_logger().info('Debug info activated')
            print('Single detection: %d' % (self._single_human))
            print('Threshold for the detection: %d' % (self._threshold_human))
            #print('Dimensions of the image: %d - %d' % (self._width_img, self._height_img))
            print('Topic name for the image: %s' % (self._topic_image_name))
            print('Topic name for the humans: %s' % (self._topic_humans_name))
            ## rospy.loginfo('Topic name for the humans frame: %s', self._topic_human_frame_name)


    def vel_human(self, current_center, prev_center):
        vel = [0.0, 0.0]
        vel[0] = current_center[0] - prev_center[0]
        vel[1] = current_center[1] - prev_center[1]
        return vel

        ## no use


    def update_window(self, w_frame, h_frame, initial_window, center, vel):
        new_window = initial_window
        new_center = center
        width = initial_window[1]
        height = initial_window[3]
        new_center = [center[0] + vel[0], center[1] + vel[1]]
        new_window[1] = width + 10
        new_window[3] = height + 10
        new_window[0] = int(new_center[0] - new_window[1] / 2.0)
        new_window[2] = int(new_center[1] - new_window[3] / 2.0)
      

        for i in range (4):
            if new_window[i] < 0:
                new_window[i] = 0

        if new_window[0] + new_window[1] > w_frame:
            new_window[1] = w_frame - new_window[0]
        if new_window[2] + new_window[3] > h_frame:
            new_window[3] = h_frame - new_window[2]

        return [new_window, new_center]


    def humans_to_msg(self, humans, frame, cloud):
        humanArray_msg = HumanArray2() ## return array
        ## data about the image
        (rows, cols, chan) = frame.shape
        humanArray_msg.image_w = cols
        humanArray_msg.image_h = rows

        ##Image field
        ## header for the msg == header image
        humanArray_msg.header = self._header_ros
        new_image_msg = ImageDepthHuman()
        try:
            new_image_msg.image_2d = self._bridge.cv2_to_imgmsg(frame, "bgr8")
            new_image_msg.image_2d.header = self._header_ros
            new_image_msg.point_cloud_3d = cloud
            new_image_msg.valid_depth = self._compute_depth
            new_image_msg.detection_active = self._move_robot

            humanArray_msg.image_human = new_image_msg
        except CvBridgeError as e:
            pass

        ##humans
        ## how many humans depens on the self._single_human var
        n_humans = 0 ## intiial = 0

        if self._single_human:
            n_humans = 1 ## we just looked for one human
            ## JUST the first one
        else:
            n_humans = len(humans)
            ## we looked for all of them
        self._vel_window = [1.0, 0.0]

        if len(humans) == 0:
            ##move robot until find humans
            self._k += 1
            if self._k > 20:
                self._k = 0
                # try:
                #     resp = self._srv_no_human([1.0, 0.0])                     # el try ... comentado por mi
                # except rospy.ServiceException as e:
                #     pass

        elif len(humans) >= n_humans:
            for k in range(n_humans):
                human_msg = self.single_human_msg(humans[k])
                if human_msg.certainty > self._threshold_human:
                    ## human is correct
                    humanArray_msg.humans.append(human_msg)
            if len(humanArray_msg.humans) == 0:
                self._k += 1
                if self._k > 20:
                    # try:                                                      # el try ... comentado por mi
                    #     resp = self._srv_no_human([1.0, 0.0])
                    # except rospy.ServiceException as e:
                    #     pass
                    self._k = 0

                
        return humanArray_msg


    def single_human_msg(self, human):
        human_msg = Human()
        part_scores = []
        ### x_values = []
        ### y_values = []
        for body_part in human.body_parts.values():
            bodyPart_msg = BodyPart()
            bodyPart_msg.idx = body_part.part_idx
            bodyPart_msg.x_percent = body_part.x
            bodyPart_msg.y_percent = body_part.y
            bodyPart_msg.score = body_part.score
            ##if self._single_human and not self._low_frec:
            ##    pass ## LATER
            part_scores.append(body_part.score)
            ### x_values.append(body_part.x)
            ### y_values.append(body_part.y)

            human_msg.parts.append(bodyPart_msg)
            ## if speech, here !!!!! do sth to change x,y to the whole
            ## image!!!
            
        human_msg.certainty = np.mean(part_scores)/10
        '''if self._single_human and human_msg.certainty > self._threshold_human:
        	if x_values is not None and y_values is not None:
	            new_user = [np.mean(x_values), np.mean(y_values)]
	            self._vel_window = self.vel_human(new_user, self._prev_user)
	            self._prev_user = new_user'''
        return human_msg


    def detect_humans(self):

        ## self._pub_image = rospy.Publisher(self._topic_human_frame_name, Image, queue_size=1)
        ## create the estimator
        #self._srv_no_human = rclpy.ServiceProxy(self._no_human_srv, NoHuman) # el create service es cuando voy adefinirlo con callback

        ##self._frame_cv = self._bridge.imgmsg_to_cv2(self._frame_ros.image_2d, "bgr8")
        ## this rate in unreal, we cannot achieve it!!
        r = self.create_rate(500) #500 Hz
        self._prev_user = [0.0, 0.0]

        thread = threading.Thread(target = rclpy.spin,args = (self,), daemon=True)
        thread.start()

        self.get_logger().info('ESTOY EN DETECT_HUMANS')

        while (rclpy.ok()):          # ANTES PONIA while(not rospy.is_shutdown())    
            #self.get_logger().info('BEFORE SLEEP')
            r.sleep()
            #self.get_logger().info('AFTER SLEEP')

        ###self._bag.close()
        rclpy.shutdown("Exiting openpose node.\n")              # ANTES PONIA rospy.signal_shutdown("Exiting openpose node.\n")
        sys.exit(1)


    def callback_image(self, frame_ros):
    	
        self.get_logger().info('ESTOY EN CALLBACK_IMAGE')
        
        try:
            self._header_ros = frame_ros.image_2d.header
            self._frame_cv = self._bridge.imgmsg_to_cv2(frame_ros.image_2d, "bgr8")
            self._cloud = frame_ros.point_cloud_3d
            self._compute_depth = frame_ros.valid_depth
            self._move_robot = frame_ros.detection_active
            self._img_ready = True
            self.publish_human_pose()
            self._img_ready = False
        except CvBridgeError as e:
            pass
        
        
    def publish_human_pose(self):
        ## copy the image into another var
        self._frame_humans = cv2.copyMakeBorder(self._frame_cv, 0, 0, 0, 0, cv2.BORDER_REPLICATE)   
        ## where are the humans ?!?!
        
        humans, times = self._openpose.inference(self._frame_humans)
        ##self._bag_time.write('times_inference', times[0])

        #if self._debug:
        #    self.get_logger().info("Inference times: %d - %d - %d ", times[0], times[1], times[2])

        ## convert to ros msg
        humanArray_msg = self.humans_to_msg(humans, self._frame_humans, self._cloud)
        
       
        ## publish human array
        self._pub_human.publish(humanArray_msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        x = OpenposeClass()
        x.detect_humans()
    except ROSInterruptException: 
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()