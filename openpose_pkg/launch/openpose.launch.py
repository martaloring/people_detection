import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('openpose_pkg'),
        'config',
        'openpose.yaml'
        )
        
    node=Node(
        package = 'openpose_pkg',
        name = 'openpose_new',
        executable = 'openpose_new',
        parameters = [config]
    )
    ld.add_action(node)
    return ld
