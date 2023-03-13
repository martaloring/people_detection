import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('openpose_pkg'),
        'config',
        'openpose.yaml'
        )
        
    astra_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('astra_camera'), 'launch'),
         '/astra_mini.launch.py'])
      )
    return LaunchDescription([
        # ASTRA
        astra_launch,
        
        # CAMERA RESIZING AND FRECUENCY ADJUSTMENT     
        Node(
            package = 'openpose_pkg',
            name = 'camera_basic',
            executable = 'camera_basic',
            output = 'screen',
            parameters = [config]),

        # PROC HUMAN
        Node(
            package = 'openpose_pkg',
            name = 'proc_human_node',
            executable = 'proc_human_node',
            parameters = [config]),
        
        # PROC DEPTH
        Node(
            package = 'openpose_pkg',
            name = 'proc_depth_node',
            executable = 'proc_depth_node',
            parameters = [config])   
        
    ])
