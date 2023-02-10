#!/usr/bin/env python
#modify: 13/07/2018
#mercedes
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import sys
from rclpy.exceptions import ROSInterruptException

class show_frame_human_class(Node):
    def __init__(self):
        super().__init__('window_node')
        ###############################
        ####    PARAMETERS         ####
        ###############################
        self.declare_parameter('~ROSTopics/video_humans_drawn_topic', '/frame_humans')
        self.declare_parameter('~DisplayImages/show_frame', False)
        self.declare_parameter('~DebugInfo/debug_info', False)
        self._topic_human_frame_drawn_name = self.get_parameter('~ROSTopics/video_humans_drawn_topic').get_parameter_value().string_value
        self._show_image = self.get_parameter('~DisplayImages/show_frame').get_parameter_value().bool_value
        self._debug = self.get_parameter('~DebugInfo/debug_info').get_parameter_value().bool_value

        self._img_ready = False

        ######################
        #### OPENCV BRIDGE ###
        ######################
        self._bridge = CvBridge()

    def callback_img(self, frame_ros):
        try:
            self._frame_cv = self._bridge.imgmsg_to_cv2(frame_ros, "bgr8")
            self._img_ready = True
        except CvBridgeError as e:
            print (e)

    def show_frame_loop(self):

        r = self.create_rate(500) #500 Hz
        self._sub_img = self.create_subscription(Image, self._topic_human_frame_drawn_name, self.callback_img)
        while (not rclpy.ok()):
            #3if the user wants to show the image
            if (self._show_image):
                #if there is a new image
                if (self._img_ready):
                    self._img_ready = False 
                    cv2.imshow("human frames", self._frame_cv)
                    cv2.waitKey(1)
            r.sleep()
        cv2.destroyAllWindows()
        rclpy.shutdown("Closing window")
        sys.exit(1)

############################
def main(args=None):
    rclpy.init(args=args)

    try:
        window = show_frame_human_class()
        window.show_frame_loop()
    except ROSInterruptException: 
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
