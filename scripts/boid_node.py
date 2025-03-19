#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, OccupancyGrid, Path
import sys
import numpy as np
import re
from boid import Boid
from visualization_msgs.msg import Marker, MarkerArray
from utils.OccupancyMap import OccupancyMap


class BoidNode:
    def __init__(self,id,boid_count):
        rospy.init_node('boid_node', anonymous=True)

        self.id = id
        self.frame_id = '/robot_{}/odom'.format(id)
        self.boid_count = boid_count
        self.map = OccupancyMap() 
        self.trajectory = Path()

        self.enable_visualization = True
        self.visualize_array = MarkerArray()
        self.visualize_acc_array = MarkerArray()


        self.boid = Boid()
        self.vel_pub = rospy.Publisher('robot_{}/cmd_vel'.format(self.id), Twist, queue_size=10)
        self.visual_pub = rospy.Publisher('robot_{}/visualization'.format(self.id), MarkerArray, queue_size=10)
        self.visual_pub_acc = rospy.Publisher('robot_{}/acc'.format(self.id), MarkerArray, queue_size=10)

        self.trajectory_pubs = rospy.Publisher('robot_{}/trajectory'.format(self.id), Path, queue_size=10)
        
        rospy.Subscriber('boid_goal', PoseStamped, self.goal_cb)
        self.map_subscriber = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)

        self.other_boids = []
        for i in range(boid_count):
            self.other_boids.append(Boid())
            rospy.Subscriber('robot_{}/odom'.format(i), Odometry, self.boid_cb)

        self.other_boids[self.id] = None
        self.acc_list = []

        self.dt = 0.1

        rospy.sleep(2)

        rospy.Timer(rospy.Duration(self.dt), self.run)
        rospy.Timer(rospy.Duration(self.dt), self.visualize)


    def boid_cb(self, msg):
        if msg.header.frame_id == self.frame_id:
            # update current state of the boid
            self.boid.position = msg.pose.pose.position
            self.boid.velocity.x = msg.twist.twist.linear.x
            self.boid.velocity.y = msg.twist.twist.linear.y
            self.boid.heading = np.arctan2(self.boid.velocity.y ,self.boid.velocity.x)

        else:
            id = int(re.findall(r'\d+', msg.header.frame_id)[0])
    
            self.other_boids[id].position = msg.pose.pose.position
            self.other_boids[id].velocity.x = msg.twist.twist.linear.x
            self.other_boids[id].velocity.y = msg.twist.twist.linear.y
            if msg.twist.twist.linear.x != 0.0 and msg.twist.twist.linear.y != 0.0: # preserve current heading 
                self.other_boids[id].heading = np.arctan2(self.other_boids[id].velocity.y, self.other_boids[id].velocity.x)

            

    def map_cb(self, gridmap):
        env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
        # Set avoid obstacles - The steer to avoid behavior (IN THE DICTIONARY) requires the map, resolution, and origin
        self.map.set(data=env, 
                    resolution=gridmap.info.resolution, 
                    origin=[gridmap.info.origin.position.x, gridmap.info.origin.position.y])
            

    def goal_cb(self, msg):
        if msg.header.frame_id == self.frame_id: 
            # rospy.logwarn("Goal recieved")
            self.boid.goal = msg.pose.position


    def run(self,_):
        self.boid.update_perception_field(self.map)

        self.boid.update_neigbors(self.other_boids)

        nav_acc = self.boid.navigation_acc()
        sep_acc = self.boid.seperation_acc()
        coh_acc = self.boid.cohesion_acc()
        align_acc = self.boid.allignment_acc()
        obs_acc = self.boid.obstacle_acc()
        
        zero = Point()
        self.acc_list = [nav_acc,sep_acc,coh_acc,align_acc,obs_acc]
        # self.acc_list = [zero,zero,zero,zero,obs_acc] #  for report demonstration

        all_acc = self.boid.combine_acc_priority(nav_acc,sep_acc,coh_acc,align_acc,obs_acc)    
        # all_acc = self.boid.combine_acc_priority(nav_acc,sep_acc,zero,zero,obs_acc) #  for report demonstration       
    
        out_vel = self.boid.cal_velocity(all_acc,self.dt)

        cmd_vel = Twist()
        cmd_vel.linear.x = out_vel.x
        cmd_vel.linear.y = out_vel.y

        self.vel_pub.publish(cmd_vel)

    ################################################
    #### Visualization
    ################################################
    def visualize(self,_):
        if self.enable_visualization:
            self.visualize_neighbor()
            self.visualize_goal()
            self.visualize_acc()
            self.update_trajectory()
            
            self.visual_pub.publish(self.visualize_array)
            self.visual_pub_acc.publish(self.visualize_acc_array)

    def visualize_neighbor(self):
        marker_id  = 0
        scale = [0.15, 0.15, 0.15]
        color = [1,0,0,1]
        frame_id = "map"
        ns = "neighbors"

        marker = self.create_marker(999,ns, Marker.SPHERE, [self.boid.position.x,self.boid.position.y,0.2], 
            [0.1,0.1,0.1], [0,0,1,1], frame_id, None)
        self.visualize_array.markers.append(marker)


        # for n in self.boid.neighbor_boids:
        #     marker = self.create_marker(marker_id,ns, Marker.SPHERE, [n.position.x, n.position.y, 0.2], 
        #     scale, color, frame_id,None)
        #     marker_id += 1

        #     self.visualize_array.markers.append(marker)
        
    
    def visualize_acc(self):
        marker_id  = 30
        scale = [0.05, 0.1, 0.0] # shaft, head
        colors = [
                [1, 0, 0, 1],  # Red - nav
                [0, 1, 0, 1],  # Green -sep
                [0, 0, 1, 1],  # Blue  - coh
                [1, 1, 0, 1],  # Yellow - allign
                [1, 0, 1, 1]   # Magenta - obs
            ]
        frame_id = "robot_{}/base_footprint".format(self.id)
        ns = "acc"
        
        for i, acc in enumerate(self.acc_list):
            points = [Point(0,0,0),Point(acc.x, acc.y,0)]
            color = colors[i]
            marker = self.create_marker(marker_id,ns, Marker.ARROW, [0,0,0.0], 
                scale, color, frame_id, points)
            marker_id += 1
            self.visualize_acc_array.markers.append(marker)

    def update_trajectory(self):
        # Create a new pose stamped with the current position
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position = self.boid.position

        # Update the trajectory for the specified robot
        
        self.trajectory.poses.append(pose_stamped)

        traj_to_keep = 500
        if len(self.trajectory.poses) > traj_to_keep:
            self.trajectory.poses = self.trajectory.poses[-traj_to_keep:]

        self.trajectory.header.frame_id = "map"

        self.trajectory_pubs.publish(self.trajectory)

        

    def visualize_goal(self):
        if self.boid.goal:
            frame_id = "map"
            ns = "goal"

            marker = self.create_marker(666,ns, Marker.SPHERE, [self.boid.goal.x,self.boid.goal.y,0.2], 
                [0.3,0.3,0.3], [1,0,0,1], frame_id, None)
            self.visualize_array.markers.append(marker)

    def create_marker(self, marker_id, ns, marker_type, position, scale, color, frame_id,points):
        marker = Marker()
        marker.header.frame_id = frame_id  # Reference frame (change as necessary)
        marker.header.stamp = rospy.Time.now()

        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD

        # Set marker position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set maker Points
        if points:
            marker.points = points

        # Set marker scale
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        # Set marker color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]  # Alpha (transparency)

        return marker
      

    
        
    

if __name__ == '__main__':
    try:
        id = int(sys.argv[1])
        boid_count = rospy.get_param('boid_count', 5)

        boid_node = BoidNode(id,boid_count)

        rospy.loginfo('Running Boid Node ID: %d' % id)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass