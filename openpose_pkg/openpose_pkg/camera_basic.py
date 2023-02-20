#!/usr/bin/env python

import rclpy
import sys
sys.path.append('/home/mapir/ros2_ws/src/openpose_pkg/openpose_pkg')
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int16
import sys
import numpy as np
from openpose_interfaces.srv import *
from openpose_interfaces.msg import *
# from calibrator import Calibrator
import rospkg #for the ros pkg path
import time
from rclpy.exceptions import ROSInterruptException
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

#############################################################################
######### this node resizes the images AND publishes it with different Hz ###
#############################################################################
class camera_usb(Node):
    def __init__(self):
        super().__init__('camera_basic')

        self._bridge = CvBridge() ## cv bridge

        ###############################
        ####    PARAMETERS         ####  por lo que he visto, para que las variables sea, bool, int... hay q hacer lo de value,
        ###############################  si no se pone lo de value, las variables quedan como parametros¿? structs¿?
        ## debug flag. show info
        self.declare_parameter('~DebugInfo/debug_info', True)
        self._debug = self.get_parameter('~DebugInfo/debug_info').get_parameter_value().bool_value
        
        ## dim of the image
        self.declare_parameter('~ImageParameters/factor', 1.0)
        self._factor_dim = self.get_parameter('~ImageParameters/factor').get_parameter_value().double_value
        
        ## frecuencies
        self.declare_parameter('~FrecInference/off_frec', 0)
        self.declare_parameter('~FrecInference/low_frec', 5)
        self.declare_parameter('~FrecInference/high_frec', 10)
        self._frec_off = self.get_parameter('~FrecInference/off_frec').get_parameter_value().integer_value
        self._frec_low = self.get_parameter('~FrecInference/low_frec').get_parameter_value().integer_value
        self._frec_high = self.get_parameter('~FrecInference/high_frec').get_parameter_value().integer_value

        ## services names
        self.declare_parameter('~ROSServices/change_frec_srv', '/change_frec')
        self.declare_parameter('~ROSServices/start_detection_srv', '/openpose/start_detection_humans_service')
        self._srv_change_name = self.get_parameter('~ROSServices/change_frec_srv').get_parameter_value().string_value
        self._srv_start_name = self.get_parameter('~ROSServices/start_detection_srv').get_parameter_value().string_value


        ## topics names
        self.declare_parameter('~ROSTopics/image_topic', 'openpose/usb_cam/image_dim_CODE') ##pub -- esto es lo que le pasa al siguente nodo (openpose_new)
        #self.declare_parameter('~ROSTopics/image_topic', None) ##pub -- esto es lo que le pasa al siguente nodo (openpose_new)
        #self.declare_parameter('~ROSTopics/image_cam_topic', '/image_raw3') ##sub
        self.declare_parameter('~ROSTopics/depth_cloud_topic', '/camera/depth/points') ##sub -- depth cloud?
        self.declare_parameter('~ROSTopics/rgb_cam_topic', '/camera/color/image_raw') ##sub -- aqui tenemos que publicar desde la camara
        self._topic_image =  self.get_parameter('~ROSTopics/image_topic').get_parameter_value().string_value
        #self._topic_image_raw = self.get_parameter('~ROSTopics/image_cam_topic').get_parameter_value().string_value
        self._topic_point_cloud_name = self.get_parameter('~ROSTopics/depth_cloud_topic').get_parameter_value().string_value
        self._topic_rgb_image = self.get_parameter('~ROSTopics/rgb_cam_topic').get_parameter_value().string_value

        #####################
        ### DEBUG INFO    ###
        #####################
   
        if self._debug:
            self.get_logger().info('Debug info activated')
            print('Off frecuency: %d' % (self._frec_off))
            print('Low frecuency: %d' % (self._frec_low))
            print('High frecuency: %d' % (self._frec_high))
            print('Factor for the image: %d' % (self._factor_dim))
            print('RGB Topic name: %s' % (self._topic_rgb_image))
            print('Topic name for the image: %s' % (self._topic_image))

        self._zero_frec_off = False ## indicates if the off frec is zero, aka, do not publish

        if self._frec_low < 1.0:
            self._frec_low = 1.0

        if self._frec_high < 1.0:
            self._frec_high = 1.0

        if self._frec_high < self._frec_low:
            self._frec_high = self._frec_low + 1.0

        if self._frec_off <= 0.0:
            self._frec_off = 1.0 ##so that rate does not explode
            self._zero_frec_off = True

        ## useful vars
        self._mode_off = True

        self._img_ready = False
        self._frame_cv1 = None
        self._header_img = None ## we need to add this

        self._flag_cloud = False
        self._cloud_points = None

        self._time_cloud = 0.0
        self._time_img = 0.0

        self._detection_active = False
        self._mode_cam = True ## false = fish cam, true = rgbd cam    he puesto la rgbd, estaba la fish

        ####################
        ##  CALIBRATION  ###
        ####################
        # reading calibration info
        # rospack = rospkg.RosPack()
        # self._pkg_path = rospack.get_path('openpose_pkg') + '/calibration/'
        # self._K_path = self._pkg_path + 'K.npy'
        # self._dist_path = self._pkg_path + 'dist.npy'
        # self._K = np.load(self._K_path)
        # self._dist = np.load(self._dist_path)
        # self._calibrator = Calibrator(self._K, self._dist)

        
        ## services
        self._change_det_srv = self.create_service(ChangeFrecDetection, self._srv_change_name, self.change_frec_srv)
        self._start_det_srv = self.create_service(StartDetectionHuman, self._srv_start_name, self.start_det_srv_callback)


        ## topics and qos
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._sub_point_depth = self.create_subscription(PointCloud2, self._topic_point_cloud_name, self.callback_points, qos_profile)

        self._sub_cam_rgb = self.create_subscription(Image, self._topic_rgb_image, self.callback_image_rgb, qos_profile) ## rgb-d cam

        self._pub_img = self.create_publisher(ImageDepthHuman, self._topic_image, 1)

        self._r = self.create_rate(self._frec_off) # so we start with the off frecuency, 

    def start_det_srv_callback(self, req, res):
        ##this service start the detection
        self._detection_active = False
        if self._debug:
            self.get_logger().info('Start Detection callback!!!')

        ## compute depth
        if req.compute_depth == 'on':
            self._detection_active = True
            self.get_logger().info('DETECTION ON')
        else:
            self._detection_active = False
            self._mode_cam = True
            self.get_logger().info('DETECTION OFF')

            
        if req.initial_frec == 'low':
            self.get_logger().info('Lowfrec')
            self._r = self.create_rate(self._frec_low)
            self._mode_off = False

        elif req.initial_frec == 'off':
            self._detection_active = False
            self._mode_cam = True
            self.get_logger().info('Offfrec')
            self._r = self.create_rate(self._frec_off)
            self._mode_off = True

        else:
            self.get_logger().info('Highfrec')
            self._r = self.create_rate(self._frec_high)
            self._mode_off = False
        
        return res

    def change_frec_srv(self,req, res):
        if req.change_frec_to == 'off':
            ## we start
            self._r = self.create_rate(self._frec_off)
            self._mode_off = True

            self._detection_active = False
            self._mode_cam = True ##change to fish-eye cam     -- lo pongpo a true
            self.get_logger().info('Frecuencia OFF')
            
        elif req.change_frec_to == 'low':
            
            self._mode_off = False
            self._r = self.create_rate(self._frec_low)
            self.get_logger().info('Frecuencia puesta a LOW')
            
        elif req.change_frec_to == 'high':
            
            self._mode_off = False
            self._r = self.create_rate(self._frec_high)
            self.get_logger().info('Frecuencia puesta a HIGH')
            
        return res                                                                    # ANTES ERA: return ChangeFrecDetectionResponse()
    
    def callback_points(self, point_cloud):
        self._cloud_points = point_cloud
        self._flag_cloud = True
        self._time_cloud = time.time()

    def callback_image(self, frame_cam):

        print("nada")

        ##this one is coming form the usbcam (aka fish)
        # if not self._mode_cam: ##if fish
        #     try:
        #         if self._debug:
        #             self.get_logger().info('Fish-eye cam callback')
                   
        #         self._header_img = frame_cam.header
        #         self._frame_cv1 = self._bridge.imgmsg_to_cv2(frame_cam, "bgr8")
        #         self._img_ready = True
                
        #     except CvBridgeError as e:
        #         print(e)    

    def callback_image_rgb(self, frame_cam):
        if self._mode_cam == True:
            try:
                if self._debug:
                    self.get_logger().info('RGB-D Callback')
                self._header_img = frame_cam.header
                self._frame_cv1 = self._bridge.imgmsg_to_cv2(frame_cam, "bgr8")
                self._img_ready = True
                self._time_img = time.time()
                
            except CvBridgeError as e:
                print(e) 

    def undistort_resize(self, frame):
        frame_resize = None
        # if self._debug:
        #     self.get_logger().info('undistort image')
        if frame is not None:
            frame_undis= cv2.copyMakeBorder(frame,0,0,0,0,cv2.BORDER_REPLICATE)

            if frame_undis is not None:
                ## apply factor
                (rows, cols, chan) = frame_undis.shape
                if(self._factor_dim > 0.0):
                    frame_resize = cv2.resize(frame_undis, (int(cols*self._factor_dim), int(rows*self._factor_dim)), interpolation=cv2.INTER_CUBIC)

                else:
                    if (int(-cols/self._factor_dim) < 50 or int(-rows/self._factor_dim) < 50):
                        if self._debug:
                            self.get_logger().info("Error, new dimensions for the image too small!!!. factor = 1.0")
                        self._factor_dim = -1

                    frame_resize = cv2.resize(frame_undis, (int(-cols/self._factor_dim), int(-rows/self._factor_dim)), interpolation=cv2.INTER_CUBIC)
                        
        return frame_resize

    def send_images_loop(self): #timer_callback

        self._detection_active = False
        thread = threading.Thread(target = rclpy.spin,args = (self,), daemon=True)
        thread.start()
        
        while (rclpy.ok()):

            if not self._detection_active and self._img_ready: ## we are using the fish - no tiene por que
                
                if self._debug:
                    self.get_logger().info('Sending image without detection')

                if self._mode_off and self._zero_frec_off:
                    pass
                else:
                    image_msg = ImageDepthHuman()
                    try:
                        new_frame = self.undistort_resize(self._frame_cv1)
                        if new_frame is not None:
                            image_msg.image_2d = self._bridge.cv2_to_imgmsg(new_frame, "bgr8")
                            image_msg.image_2d.header = self._header_img

                            image_msg.point_cloud_3d = PointCloud2()
                            image_msg.valid_depth = 0 
                            image_msg.detection_active = self._detection_active
                            self._pub_img.publish(image_msg)
                    except CvBridgeError as e:
                        pass

                self._img_ready = False

            elif self._img_ready and self._detection_active:
                if self._debug:
                    self.get_logger().info('Detection active, sending image')
               
                ## undistort and resize image 
                new_frame = self.undistort_resize(self._frame_cv1)
                if new_frame is not None:
                    image_msg = ImageDepthHuman()
                    try:
                        image_msg.image_2d = self._bridge.cv2_to_imgmsg(new_frame, "bgr8")
                        image_msg.image_2d.header = self._header_img

                        image_msg.valid_depth = int(self._mode_cam)
                        image_msg.detection_active = self._detection_active
                        
                        if not self._mode_cam:
                            image_msg.point_cloud_3d = PointCloud2()                            # por que vuelve a comprobar el modo? lo de las flags ni idea de lo que es
                            
                        elif self._mode_cam and self._flag_cloud:
                            image_msg.point_cloud_3d = self._cloud_points
                            image_msg.point_cloud_3d.header.frame_id = 'camera_color_optical_frame'


                        self._pub_img.publish(image_msg)
                        self._img_ready = False
                        self._flag_cloud = False

                        if self._debug:
                            self.get_logger().info('Image sent with detection')

                    except CvBridgeError as e:
                        pass
            self._r.sleep()
            
                
        rclpy.shutdown("Closing camera node.\n")
        sys.exit(1)
        
def main(args=None):
    rclpy.init(args=args)

    try:
        x = camera_usb()
        x.send_images_loop()
    except ROSInterruptException: 
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
