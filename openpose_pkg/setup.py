import os
from glob import glob
from setuptools import setup

package_name = 'openpose_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        #Include all launch files from type python
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mapir',
    maintainer_email='mapir@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_basic = openpose_pkg.camera_basic:main',
            'proc_depth_node = openpose_pkg.proc_depth_node:main',
            'openpose_new = openpose_pkg.openpose_new:main',
            'User2Person2 = openpose_pkg.User2Person2:main',
            'proc_human_node = openpose_pkg.proc_human_node:main',
            'window2 = openpose_pkg.window2:main',
        ],
    },
)
