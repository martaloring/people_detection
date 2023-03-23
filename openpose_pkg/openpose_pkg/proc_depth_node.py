#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray
from cv_bridge import CvBridge
import sys
from openpose_interfaces.msg import *
sys.path.append('/home/mapir/ros2_ws/src/openpose_pkg/openpose_pkg')
from proc_depth import *
from rclpy.exceptions import ROSInterruptException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_geometry_msgs
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class HumanDepthProcessorClass(Node):
    def __init__(self):
        super().__init__('proc_human_depth_2d_node')

        ###############################
        ####    PARAMETERS         ####
        ###############################
        ## debug flag. show info
        self.declare_parameter('DebugInfo.debug_info', True)
        self._debug = self.get_parameter('DebugInfo.debug_info').get_parameter_value().bool_value

        self.declare_parameter('ImageParameters.factor', 0.2)
        self._factor = self.get_parameter('ImageParameters.factor').get_parameter_value().double_value

        self.declare_parameter('HumanDetected.compute_body', False)
        self._create_body_3d = self.get_parameter('HumanDetected.compute_body').get_parameter_value().bool_value

        ## topics names
        self.declare_parameter('ROSTopics.humans_3d_topic', '/users') ##sub
        self.declare_parameter('ROSTopics.cloud_topic', '/cloud_topic') ##sub
        self.declare_parameter('ROSTopics.users_3d_topic', '/users3D') ##pub
        self.declare_parameter('ROSTopics.poses_topic', '/poses_topic') ##pub       
        #self.declare_parameter('ROSTopics.marker_users_topic', '/markers_users') # array de markers en el espacio, cada uno marca una pose
        #self.declare_parameter('ROSServices.no_human_srv', '/human')
        #self.declare_parameter('ROSTopics.markers_3d_body_parts_topic', '/markers_body') # array de markers en el espacio, indicando las partes del humano en el espacio (lineas y esferas)

        self._topic_human_3d_name = self.get_parameter('ROSTopics.humans_3d_topic').get_parameter_value().string_value ##sub
        self._topic_cloud = self.get_parameter('ROSTopics.cloud_topic').get_parameter_value().string_value ##sub
        self._topic_users_3d = self.get_parameter('ROSTopics.users_3d_topic').get_parameter_value().string_value ##pub
        self._topic_poses = self.get_parameter('ROSTopics.poses_topic').get_parameter_value().string_value ##pub
        #self._topic_marker = self.get_parameter('ROSTopics.marker_users_topic').get_parameter_value().string_value
        #self._no_human_srv = self.get_parameter('ROSServices.no_human_srv').get_parameter_value().string_value
        #self._topic_body_3d_markers = self.get_parameter('ROSTopics.markers_3d_body_parts_topic').get_parameter_value().string_value

        #publishers and subscribers
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_profile2 = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

        self._sub_humans = self.create_subscription(UserRGBDArray, self._topic_human_3d_name, self.callback_users, qos_profile) ## users from proc. image node. # array de UserRGBD (para cada usuario: nombre, coord. cara en img, pose, body parts...)
        self._sub_cloud = self.create_subscription(PointCloud2, self._topic_cloud, self.callback_cloud, qos_profile) 

        self._pub_user = self.create_publisher(User3DArray, self._topic_users_3d, qos_profile) ## array de User3D (para cada usuario: nombre, pose, body parts, altura...)
        self._pub_poses= self.create_publisher(PoseArray, self._topic_poses, qos_profile2) ##final poses
        #self._pub_marker = self.create_publisher(MarkerArray, self._topic_marker, qos_profile) 
        #self._pub_markers_body = self.create_publisher(MarkerArray, self._topic_body_3d_markers, qos_profile) ##img with drawn human


        self._header_users = None
        self._users = None
        self._width_img = 100.0
        self._height_img = 100.0
        self._cloud = PointCloud2()

        # camera to base_link params
        self.declare_parameter('CameraToBaseLinkTF.translation_x', 0.15)
        self.declare_parameter('CameraToBaseLinkTF.translation_y', 0.0)
        self.declare_parameter('CameraToBaseLinkTF.translation_z', 1.0)
        self.declare_parameter('CameraToBaseLinkTF.angle_x', 0.0)
        self.declare_parameter('CameraToBaseLinkTF.angle_y', 0.0)
        self.declare_parameter('CameraToBaseLinkTF.angle_z', 0.0)
        self._translation_x = self.get_parameter('CameraToBaseLinkTF.translation_x').get_parameter_value().double_value
        self._translation_y = self.get_parameter('CameraToBaseLinkTF.translation_y').get_parameter_value().double_value
        self._translation_z = self.get_parameter('CameraToBaseLinkTF.translation_z').get_parameter_value().double_value
        self._rotation_x = self.get_parameter('CameraToBaseLinkTF.angle_x').get_parameter_value().double_value
        self._rotation_y = self.get_parameter('CameraToBaseLinkTF.angle_y').get_parameter_value().double_value
        self._rotation_z = self.get_parameter('CameraToBaseLinkTF.angle_z').get_parameter_value().double_value

        self.declare_parameter('CameraToBaseLinkTF.general_camera_frame', 'camera_link')
        self._general_camera_frame = self.get_parameter('CameraToBaseLinkTF.general_camera_frame').get_parameter_value().string_value
        self.declare_parameter('CameraToBaseLinkTF.depth_cloud_frame', 'camera_color_optical_frame')
        self._depth_cloud_frame = self.get_parameter('CameraToBaseLinkTF.depth_cloud_frame').get_parameter_value().string_value

        # CREAMOS LA TRANSFORMADA (camera_link -> base_link) PARA ENVIAR LA POSE RESPECTO A BASE_LINK

        self.t = TransformStamped()
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.t.header.stamp = self.get_clock().now().to_msg()
        self.t.header.frame_id = 'base_link'
        self.t.child_frame_id = self._general_camera_frame

        self.t.transform.translation.x = self._translation_x
        self.t.transform.translation.y = self._translation_y
        self.t.transform.translation.z = self._translation_z
        quat = quaternion_from_euler(
            self._rotation_x, self._rotation_y, self._rotation_z)
        self.t.transform.rotation.x = quat[0]
        self.t.transform.rotation.y = quat[1]
        self.t.transform.rotation.z = quat[2]
        self.t.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(self.t)

        # CREAMOS LOS OBJETOS NECESARIOS PARA ESCUCHAR (camara->map)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        while (rclpy.ok()):
            r.sleep()

        
        rclpy._shutdown("Exiting openpose publisher node.\n")
        sys.exit(1)
    

    def callback_users(self, user_array):

        if (self._debug):
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
                if (self._debug):
                    self.get_logger().info('cloud correct')
                    print('number of users: %d' % (len(userarray_msg.users)))

                if len(userarray_msg.users) > 0:
                    #self._pub_user.publish(userarray_msg)
                    #self._pub_marker.publish(userarrarmarker_msg)
                    # if (self._debug):
                    #     self.get_logger().info('publishing users3D')

                    # if self._create_body_3d:
                    #     self._pub_markers_body.publish(markers_body)
                    
                    # TRANSFORMAMOS LAS POSES Y LAS PUBLICAMOS
                    camera_to_map = self.tf_buffer.lookup_transform(
                        'map',
                        self._depth_cloud_frame,
                        rclpy.time.Time())
                    
                    pos_msg = PoseArray()
                    pos_msg.header = userarray_msg.header

                    for i in range(0,(len(userarray_msg.users))):
                        pose_transformed = tf2_geometry_msgs.do_transform_pose(userarray_msg.users[i].pose_3d, camera_to_map)
                        pos_msg.poses.append(pose_transformed)

                    self._pub_poses.publish(pos_msg)
        else:
            pos_msg = PoseArray()
            self._pub_poses.publish(pos_msg)


    def callback_cloud(self, cloud):

        self._cloud = cloud
        if (self._debug):
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
