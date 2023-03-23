# people_detection
This ROS2 package integrates a human pose estimation system based on body parts recognition and depth points (both provided by a RGBD camera).

# Input
    - RGB frames and 3D point cloud (both must be from the same RGBD camera).
	    RGB frames:
            Type of msg: Image
            Topic name (default):  “/camera/color/image_raw”
            QoS profile: KEEP_LAST (history), BEST_EFFORT (reliability)
	    Depth points:
            Type of msg: PointCloud2
            Topic name (default): “/camera/depth/points”
            QoS profile: KEEP_LAST (history), BEST_EFFORT (reliability)

    - “map” to “base_link” transform must be sent
	
# Output
    - Poses array ('map' as a reference frame) and users array
	    Poses array:
		    Type of msg: PoseArray
            Topic name (default):  “/poses_topic”
            QoS profile: KEEP_LAST (history), RELIABLE (reliability)

        Users array:
	        Type of msg: User3DArray
            Topic name (default):  “/users3D”
            QoS profile: KEEP_LAST (history), BEST_EFFORT (reliability)

    - Drawn frames
	    Type of msg: Image
        Topic name (default):  “/openpose/frame_humans/draw_img”
        QoS profile: KEEP_LAST (history), BEST_EFFORT (reliability)

# Nodes structure
    The whole process is divided into 4 main tasks, each one implemented in a different ROS2 node:  RGBD CAMERA → ‘camera_basic’ → ‘openpose_new’ → ‘proc_human_node’ → ‘proc_depth_node’ → OUTPUT INFORMATION.

    - ‘camera_basic’ node: resizes the images by ['ImageParameters.factor’] factor and publishes them with different Hz [‘FrecInference.high_frec’] ,[‘FrecInference.low_frec’] or [‘FrecInference.off_frec’].  This node will not be sending depth points information unless ‘ROSServices.start_detection_srv’ service is called with ‘compute_depth’ input field set to ‘on’. See ‘# Parameters’ and ‘# Services’ for more information.

    - ‘openpose_new’ node: by calling the TensorFlow estimator (“estimator2.py”), it runs the inference function from the loaded “Mobilenets” NeuronalNetwork graph. Having RGB frames as a input, it generates “HumanArray2” messages (body parts information).

    - ‘proc_human_node’ node: takes the resized RGB frames and the body parts information and generates a RGB image with the skeleton parts superimposed on it (very useful for visualization). It reads the “HumanArray2” message and publishes an “UserRGBDArray” message.

    - ‘proc_depth_node’ node: publishes the final position and orientation of the humans in the map using the “UserRGBDArray” and the depth points. It also generates the “User3DArray”. It calculates “camera_link” to “map” transform (using [‘CameraToBaseLinkTF’] parameters) so final poses are referenced to “map” frame.

# Services
    - start_detection_srv: stops and starts sending depth points information. It is necessary to call this service after you launch the ‘camera_basic’ node . It has the following input fields:
        ‘initial_frec’: [high] - [low] - [off]. These frequencies are set to [‘FrecInference’] parameters values.
        ‘compute_depth’: [on] - [off]. 
        Use example, for starting detection: “ ros2 service call /openpose/start_detection_humans_service openpose_interfaces/srv/StartDetectionHuman "{initial_frec: 'high', compute_depth: 'on'}" ”

    - change_frec_srv: changes the frequency that “camera_basic” node publishes frames. It has the following input field:
        ‘change_frec_to’: [high] - [low] - [off]. These frequencies are set to [‘FrecInference’] parameters values.

# Parameters
Package parameters can be found in ‘config/openpose.yaml’ file. Every parameter is described in such file.

# Prerequisites
    - “openpose_pkg” package
    - “openpose_interfaces” package
    - “tf_transformations” package
    - “openpose_new” node launches a NN which is GPU intense, therefore it is recommended to run the node in a computer with a powerful GPU
    - TensorFlow v2 must be installed in the computer where the estimator will be run. It needs python 3.8. In case of a different python version, it is recommended to create a “virtual python environment” (easy to do with “conda”).

# Launch files
    - ‘main.launch.py’ launches the camera and every openpose node, except from “openpose_new” node (because it may be run in a different computer). It specifies the parameters file.
    - ‘openpose.launch.py’ launches “openpose_new” node and specifies the parameters file.

