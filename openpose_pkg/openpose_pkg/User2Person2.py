#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import sys
import numpy as np
from openpose_interfaces.msg import *
from rclpy.duration import Duration
from rclpy.time import Time
import time
from openpose_interfaces.msg import People, Person
from rclpy.exceptions import ROSInterruptException


class ProcUserToPerson(Node):
    def __init__(self):
        super().__init__('converter_node2')

        self.declare_parameter('~ROSTopics/users_3d_topic', '/empty_topic')
        self.declare_parameter('~ROSTopics/people', '/people')
        self._topic_users_3d = self.get_parameter('~ROSTopics/users_3d_topic').get_parameter_value().string_value
        self._topic_people = self.get_parameter('~ROSTopics/people').get_parameter_value().string_value

        self.declare_parameter('~HumanDetected/keep_alive', 0.3)
        self.declare_parameter('~HumanDetected/max_freq_pub', 10)
        self._keep_alive = self.get_parameter('~HumanDetected/keep_alive').get_parameter_value().double_value
        self._max_freq_pub = self.get_parameter('~HumanDetected/max_freq_pub').get_parameter_value().integer_value
        
        self._human_detected = False
        self._timeout_detection = False
        self._keep_alive_dur = Duration(self._keep_alive) # ANTES PONIA rospy.Duration.from_sec(self._keep_alive)
        
        self._r = self.create_rate(self._max_freq_pub)
        self._sub_user = self.create_subscription(User3DArray, self._topic_users_3d, self.callback_users)

        self._pub_people = self.create_publisher(People, self._topic_people, queue_size = 100)
        self._people_msg = People()
        self._people_former = People()
        self._started_time = time.time() #self.get_clock().now()# ANTES PONIA  rospy.Time.now()

        ##rospy.loginfo("user topic %s", self._topic_users_3d)
        ##rospy.loginfo("people topic %s", self._topic_people)

    def loop_converter(self):
       
        while (rclpy.ok()):          # ANTES PONIA while(not rospy.is_shutdown()()
            if self._human_detected:
                # we publish the new people msg
                self._human_detected = False # down the flag
                self._pub_people.publish(self._people_msg) # publish the msg
                self._started_time = time.time() # restart the timer # ANTES PONIA  rospy.Time.now() 
                self._people_former = self._people_msg # copy the msg
                #rospy.loginfo("Data from callback")
                
            elif ((time.time() - self._started_time) < self._keep_alive_dur):
                # we publish the former people msg
                if (len (self._people_msg.people) > 0):
                    self._pub_people.publish(self._people_former)
                    #rospy.loginfo("Data from past")

            else:
                self._people_former = People() # clean the msg
            
            self._r.sleep()

        ##rospy.signal_shutdown("Existing person converter")
        sys.exit(1)


    def callback_users(self, data):
        #rospy.loginfo("Data rec")
        
        self._people_msg = People() ##clean the array
        users = data.users ## this is an array
        self._people_msg.header = data.header
        for user in users:
            person_msg = Person()
            person_msg.position = user.pose_3d.position
            person_msg.reliability = user.certainty

            self._people_msg.people.append(person_msg)

        if (len (self._people_msg.people) > 0):
            self._human_detected = True # we've detected a new human
            

def main(args=None):
    rclpy.init(args=args)

    try:
        x = ProcUserToPerson()
        x.loop_converter()
    except ROSInterruptException: 
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()


