#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, OccupancyGrid
import sys
import numpy as np
import re
from boid import Boid
from visualization_msgs.msg import Marker, MarkerArray

class BoidInterface:
    def __init__(self,boid_count,leader_count):
        rospy.init_node('boid_interface', anonymous=True)
        self.leader_count = leader_count
        
        # self.neighbor_pub_list = []
        self.current_goal = None

        rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_cb)

        self.goal_pub = rospy.Publisher('boid_goal', PoseStamped, queue_size=10)


        # self.boids = []
        # for i in range(boid_count):
        #     # neighbor_pub = rospy.Publisher('boid_neighbor', PoseStamped, queue_size=10)
        #     self.neighbor_pub_list.append(neighbor_pub)
        #     rospy.Subscriber('robot_{}/odom'.format(i), Odometry, self.boid_cb)


        self.dt = 0.5

        rospy.Timer(rospy.Duration(self.dt), self.run)


    def boid_cb(self, msg):
            id = int(re.findall(r'\d+', msg.header.frame_id)[0])
    
            self.boids[id].position = msg.pose.pose.position
            self.boids[id].velocity.x = msg.twist.twist.linear.x
            self.boids[id].velocity.y = msg.twist.twist.linear.y
            self.boids[id].heading = np.arctan2(self.boids[id].velocity.y, self.boids[id].velocity.x)


    def goal_cb(self, msg):
        rospy.logwarn("Rviz Gaol Recieve")
        self.current_goal = msg.pose.position

    # def update_neigbors(self,other_boids):
    
    #     self.neighbor_boids = []  # Reset neighbor list
    #     for o_boid in other_boids:
    #         if o_boid is not None:
    #             dis = np.linalg.norm(np.array([self.position.x, self.position.y])-np.array([o_boid.position.x, o_boid.position.y]))
    #             if dis < self.neighbor_range :
    #                 self.neighbor_boids.append(o_boid)

    #     return self.neighbor_boids


    def run(self,_):
        if self.current_goal is not None:
            # publish gaol to boid leader ()
            for i in range(self.leader_count):
                goal = PoseStamped()
                goal.header.stamp = rospy.Time.now()
                goal.header.frame_id = '/robot_{}/odom'.format(i)
                goal.pose.position.x = self.current_goal.x
                goal.pose.position.y = self.current_goal.y
                self.goal_pub.publish(goal)
   

    
        
    

if __name__ == '__main__':
    try:
        boid_count = rospy.get_param('boid_count', 5)
        leader_count = rospy.get_param('leader_count', 1)

        boid_node = BoidInterface(boid_count,leader_count)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass