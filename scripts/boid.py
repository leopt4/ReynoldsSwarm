import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from random import uniform
import numpy as np


from utils.OccupancyMap import OccupancyMap
from utils.SteerToAvoid import SteerToAvoid
def wrap_angle(ang):
    return ang + (2.0 * np.pi * np.floor((np.pi - ang) / (2.0 * np.pi)))
class Boid:
    def __init__(self):
        # Initialize Boid's state
        self.position = Point()
        self.velocity = Point()
        self.heading = 0.0
        self.goal = None
        self.perception_field       = OccupancyMap()
        self.perception_radius      = 2.0
        
        # Random initial position
        self.position.x = 0
        self.position.y = 0
        
        # Random initial velocity
        self.velocity.x = 0
        self.velocity.y = 0

        #tuning params
        self.max_acc = 10.0
        self.max_vel = 0.8
        self.nav_gain = 0.8  # Navigation gain, controls the strength of the navigation behavior
        self.neighbor_range = 1.2 # 1.2
        self.neightbor_angle = np.pi/1
        self.seperation_range = 0.35
        # [10.0,0.8,1.5,0.5,2.0,0.5] #[6.3,1.5,1.5,0.5,1.4,0.5]
        # self.weight_list = [7.0,1.4,1.5,0.4,1.4,0.5]
        self.weight_list = [6.6,1.8,1.5,0.4,1.4,0.5]
        self.acc_pool = 6.65

        # weight_list = [2.0,2.0,1.5,0.5,6.3,1.5]
        # weight_list = [1.0,1.0,2.0,1.5,6.3,1.5]
        
        self.other_boids = []
        self.neighbor_boids = []


        self.obs_acc = SteerToAvoid(0.8, 0.5, 6.28) #0.6,  0.174533 # stabel [0.8,0.5]

    # def update_velocity(self, vel ):
    #     # Update velocity 
    #     self.velocity = self.limit_vel(vel)
    
    ##################################################
    #### Perception 
    ##################################################
  
    
    def update_neigbors(self,other_boids):
    
        self.neighbor_boids = []  # Reset neighbor list
        for o_boid in other_boids:
            if o_boid is not None:
                dis = np.linalg.norm(np.array([self.position.x, self.position.y])-np.array([o_boid.position.x, o_boid.position.y]))
                ang = abs(wrap_angle(self.heading - np.arctan2( o_boid.position.y - self.position.y, o_boid.position.x- self.position.x)))
                if dis < self.neighbor_range and ang < self.neightbor_angle:
                    self.neighbor_boids.append(o_boid)

        return self.neighbor_boids
        
    def update_perception_field(self, map:OccupancyMap):
        map_crop, map_dim, resolution, origin, there_is_map = map.crop_pos([self.position.x, self.position.y], self.perception_radius)
        self.perception_field.update(map_crop, map_dim, resolution, origin, there_is_map)
        # self.perception_field.show_orin_map() 
        # print(map_crop[:, 20])
        # print("______________")
        # print(map_crop[22, :])
        self.obs_acc.update_map(self.perception_field)
    
    
    ##################################################
    #### Acceleration calculation
    ##################################################

    def seperation_acc(self):
        """
        Calculate the seperation acceleration of the boid from its neighbors.
        - neighbors list contain boid in perception range [[boid#2], [boid#5],...]
        - Returns a Point() representing the seperation acceleration. (or vel not sure)
        """

        neighbor_boids = self.neighbor_boids
        sep_acc = Point()

        if neighbor_boids != []:
            # Iterate through each neighbor boid
            for n_boid in neighbor_boids:
                
                # Calculate the distance between the current boid and its neighbor
                dis = np.linalg.norm(np.array([self.position.x, self.position.y])-np.array([n_boid.position.x, n_boid.position.y]))

                # Avoid division by zero
                if dis > 0 and dis < self.seperation_range:  # Not sure if it should be exactly 0 or maybe very small value
                    # Repulsion force: inverse proportionality to the distance
                    # This creates a vector pointing away from the neighboring boid.
                    sep_acc.x += (self.position.x - n_boid.position.x)/dis**2
                    sep_acc.y += (self.position.y - n_boid.position.y)/dis**2

            # Normalize the acceleration vector
            magnitude = np.linalg.norm(np.array([sep_acc.x, sep_acc.y]))
            if magnitude > 0:
                sep_acc.x /= magnitude
                sep_acc.y /= magnitude

            return self.limit_acc(sep_acc)

        # Return zero acceleration if no neighbors are present
        return self.limit_acc(sep_acc)
        
    def cohesion_acc(self):
        """
        Calculate the seperation acceleration of the boid from its neighbors.
        - neighbors list contain boid in perception range [[boid#2], [boid#5],...]
        - Returns a Point() representing the seperation acceleration.
        """

        neighbor_boids = self.neighbor_boids
        coh_acc = Point()
        

        if neighbor_boids != []:
            avg_position = Point()
            for n_boid in neighbor_boids:
                avg_position.x += n_boid.position.x
                avg_position.y += n_boid.position.y

            avg_position.x /= len(neighbor_boids)
            avg_position.y /= len(neighbor_boids)
    
            coh_acc.x = avg_position.x - self.position.x
            coh_acc.y = avg_position.y - self.position.y

            return self.limit_acc(coh_acc)

        else:
            return self.limit_acc(coh_acc)
        
    def allignment_acc(self):
        """
        Calculate the seperation acceleration of the boid from its neighbors.
        - neighbors list contain boid in perception range [[boid#2], [boid#5],...]
        - Returns a Point() representing the seperation acceleration.
        """

        neighbor_boids = self.neighbor_boids
        allign_acc = Point()

        if neighbor_boids != []:
            # container to store average velocity of neighboring boids
            xvel_avg = 0
            yvel_avg = 0

            # loop through all neighboring boids
            for n_boid in neighbor_boids:
                # sum the velocity of neighboring boids
                xvel_avg += n_boid.velocity.x
                yvel_avg += n_boid.velocity.y

            # take the average velocity of neighboring boids as the "desired velocity"
            xvel_avg /= len(neighbor_boids)
            yvel_avg /= len(neighbor_boids)

            # compute necessary acceleration                
            allign_acc.x = (xvel_avg - self.velocity.x ) *3
            allign_acc.y = (yvel_avg - self.velocity.y ) *3
        else:
            allign_acc.x = 0
            allign_acc.y = 0
            
        return self.limit_acc(allign_acc)
        
    def obstacle_acc(self):
        boid_pose   = [0.0, 0.0]
        boid_vel    = [self.velocity.x, self.velocity.y]
        # boid_vel    = [0.8, 0.0]
        b = self.obs_acc._steer_to_avoid(boid_pose, boid_vel)
        a = 1
        combined_acc = Point()
        combined_acc.x = b[0] 
        combined_acc.y = b[1] 
        return combined_acc

    def navigation_acc(self):
        nav_acc = Point()
        nav_vel = Point()

        if self.goal:
            nav_vel.x = (self.goal.x - self.position.x) * self.nav_gain
            nav_vel.y = (self.goal.y - self.position.y) * self.nav_gain

            nav_acc.x = nav_vel.x - self.velocity.x
            nav_acc.y = nav_vel.y - self.velocity.y
        return self.limit_acc(nav_acc)
    
    def _arrival(self):
        ''' Computes required acceleration for the boid to arrive at the goal.'''
        desired_vel = Point()
        desired_acc = Point()
        desired_vel.x = 0.
        desired_vel.y = 0.
        if self.goal:
            boid_pose = [self.position.x, self.position.y] 
            boid_goal = [self.goal.x, self.goal.y, 1.0, 0.05]
            desired_velocity = np.array([0., 0.])
            if boid_goal != None:
                target_offset = np.array(boid_goal[:2]) - np.array(boid_pose[:2])   # goal[x, y], not r or tolerance. 
                distance = np.linalg.norm(target_offset)
                if distance < boid_goal[3]:
                    # print("Distance: ", distance)
                    #TODO: send a flag that boid_goal has been reached and generate a new boid_goal
                    
                    return desired_vel # Set the distance to 0 when it falls below the tolerance. 

                ramped_speed = self.max_vel * (distance / boid_goal[2])
                clipped_speed = np.minimum(ramped_speed, self.max_vel)
                desired_velocity = (clipped_speed / distance) * target_offset
                # Sample the desired velocity from the velocity space using probability
                # desired_velocity = np.array([random.uniform(-self.max_speed, self.max_speed), random.uniform(-self.max_speed, self.max_speed)]) if random.uniform(0, 1) < 0.5 else desired_velocity
            
            desired_vel.x = desired_velocity[0]*0.3
            desired_vel.y = desired_velocity[1]*0.3

            desired_acc.x = desired_vel.x - self.velocity.x
            desired_acc.y = desired_vel.y - self.velocity.y
        return desired_acc
    
    def combine_acc(self, nav_acc,sep_acc,coh_acc,allign_acc,obs_acc):
        combined_acc = Point()
        combined_acc.x = nav_acc.x  + 10*allign_acc.x + sep_acc.x + 0.2*coh_acc.x 
        combined_acc.y = nav_acc.y  + 10*allign_acc.y + sep_acc.y + 0.2*coh_acc.y 

        # combined_acc.x = nav_acc.x  
        # combined_acc.y = nav_acc.y  

        # rospy.loginfo("nav,coh,allign,sep,obs,com [x]: {},{},{},{},{}".format(nav_acc.x,coh_acc.x, allign_acc.x, sep_acc.x, obs_acc.x,combined_acc.x))
        # rospy.loginfo("nav,coh,allign,sep,obs,com [y]: {},{},{},{},{}".format(nav_acc.y,coh_acc.y, allign_acc.y, sep_acc.y, obs_acc.y,combined_acc.y))

        return combined_acc
    
    def combine_acc_priority(self, nav_acc,sep_acc,coh_acc,allign_acc,obs_acc):
        combined_acc = Point()
        arr_acc = self._arrival()
        priority_list = [obs_acc, sep_acc, arr_acc,nav_acc,allign_acc,coh_acc]
        # priority_list = [coh_acc,allign_acc, arr_acc,nav_acc, obs_acc, sep_acc]
        # priority_list = [nav_acc,arr_acc, coh_acc,allign_acc, obs_acc, sep_acc]

         

        for i in range(len(priority_list)):
            combined_acc.x += self.weight_list[i] * priority_list[i].x   
            combined_acc.y += self.weight_list[i] * priority_list[i].y  
            
            if np.linalg.norm([combined_acc.x,combined_acc.y]) > self.acc_pool: # 6.65
                # print(i)
                # print(combined_acc)
                break
        
       
        # print("----")

        return self.limit_acc(combined_acc)

        # combined_acc.x = nav_acc.x  
        # combined_acc.y = nav_acc.y  

        # rospy.loginfo("nav,coh,allign,sep,obs,com [x]: {},{},{},{},{}".format(nav_acc.x,coh_acc.x, allign_acc.x, sep_acc.x, obs_acc.x,combined_acc.x))
        # rospy.loginfo("nav,coh,allign,sep,obs,com [y]: {},{},{},{},{}".format(nav_acc.y,coh_acc.y, allign_acc.y, sep_acc.y, obs_acc.y,combined_acc.y))

        return combined_acc

    ##################################################
    #### helper functions
    ##################################################
    def cal_velocity(self, acc , dt):
        # Update velocity 
        vel = Point()
        vel.x = self.velocity.x + (acc.x*dt)
        vel.y = self.velocity.y + (acc.y*dt)


        out_vel = self.limit_vel(vel)
        return out_vel
    
    def limit_vel(self,vel):
        # Limit the vel of the boid to avoid it moving too fast
        if abs(vel.x) > self.max_vel:
            vel.x = (vel.x/abs(vel.x)) * self.max_vel
        
        if abs(vel.y) > self.max_vel:
            vel.y = (vel.y/abs(vel.y)) * self.max_vel

        return vel
    
    def limit_acc(self,acc):
        # Limit the acc of the boid to avoid it moving too fast
        if abs(acc.x) > self.max_acc:
            acc.x = (acc.x/abs(acc.x)) * self.max_acc
        
        if abs(acc.y) > self.max_acc:
            acc.y = (acc.y/abs(acc.y)) * self.max_acc

        return acc

