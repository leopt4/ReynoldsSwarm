import numpy as np
import math

from utils.OccupancyMap import OccupancyMap

class SteerToAvoid:

    # Constructor
    def __init__(self, obstacle_radius, step_angle, max_steering_angle):
        # map: 2D array of integers which categorizes world occupancy
        self.map        = OccupancyMap() 
        self.step_angle = step_angle
        self.max_steering_angle = max_steering_angle
        self.obstacle_radius=obstacle_radius
        self.steering_force = None
        self.path_valid = True
    
    #----------------------------------------------------------------behavior core functions----------------------------#
    def update_map(self, map):
        self.map = map

    def is_valid(self, point):
        # Assume grid_map is a 2D list representing the grid map
        x, y = self.map._pos_to_cell(point)
        # Check if the point is within the grid boundaries
        if 0 <= x and x < self.map.map_dim[0] and 0 <= y and y < self.map.map_dim[0]:
            # Check if the cell is free (0) or occupied (100)
            if self.map.map[x,y] == 0:
                return True  # Free cell
            else:
                return False  # Occupied cell
        else:
            return False  # Point is outside the grid boundaries

    def calc_lookahead(self, boid_x_pos, boid_y_pos, boid_theta_rad):
        """
        Calculate the lookahead point based on the current position, heading, and obstacle radius.

        Parameters:
            boid_x_pos (float): Current x-coordinate of the boid in real-world units.
            boid_y_pos (float): Current y-coordinate of the boid in real-world units.
            boid_theta_rad (float): Current heading of the boid in radians.

        Returns:
            tuple:
                - list: The map cell indices [cell_x, cell_y] of the lookahead point.
                - list: The real-world coordinates [new_x, new_y] of the lookahead point.
        """
        # Calculate the real-world coordinates of the lookahead point
        # Offset by obstacle_radius in the direction of boid_theta
        new_boid_x_pos = boid_x_pos + self.obstacle_radius * math.cos(boid_theta_rad)
        new_boid_y_pos = boid_y_pos + self.obstacle_radius * math.sin(boid_theta_rad)

        # Convert the lookahead point to map cell indices
        new_boid_cell = self.map._pos_to_cell([new_boid_x_pos, new_boid_y_pos])

        # Return the map cell indices and real-world coordinates of the lookahead point
        return new_boid_cell, [new_boid_x_pos, new_boid_y_pos]
    
    def check_path(self, p1, p2, step_size=0.15):
        """
        Check if a path between two points is valid by interpolating waypoints 
        and ensuring all waypoints are in valid positions.

        Parameters:
            p1 (list or tuple): Starting point [x, y] in real-world coordinates.
            p2 (list or tuple): Ending point [x, y] in real-world coordinates.
            step_size (float, optional): Distance between consecutive waypoints along the path. Default is 0.15.

        Returns:
            bool: True if the path is valid, False if any waypoint is invalid.
        """
        waypoints = []  # List to store the interpolated waypoints

        # Calculate the Euclidean distance between the two points
        dist = np.linalg.norm(np.array(p1) - np.array(p2))

        # Determine the number of steps needed based on the step size
        num_steps = dist / step_size
        num_steps = int(num_steps)  # Ensure num_steps is an integer

        # Generate waypoints by linear interpolation between p1 and p2
        for j in range(num_steps):
            # Calculate the interpolation factor for this step
            interpolation = float(j) / num_steps

            # Compute the interpolated x and y coordinates
            x = p1[0] * (1 - interpolation) + p2[0] * interpolation
            y = p1[1] * (1 - interpolation) + p2[1] * interpolation

            # Add the interpolated point to the waypoints list
            waypoints.append((x, y))

        # Exclude the starting point (optional, if checking only intermediate points)
        waypoints = waypoints[1:]

        # Check each waypoint for validity using the is_valid method
        for w in waypoints:
            if not self.is_valid(w):  # If any waypoint is invalid, the path is not valid
                return False

        # If all waypoints are valid, the path is valid
        return True

    def turn_back(self,boid_theta):
        return boid_theta + 3.14  # Return the opposite angle if outside the map
    
    def _get_steering_direction(self, boid_pos, boid_vel):
        """
        Determine the steering direction for a boid to avoid obstacles and maintain a valid path.

        Parameters:
            boid_pos (list or tuple): Current position of the boid [x, y] in real-world coordinates.
            boid_vel (list or tuple): Current velocity of the boid [vx, vy] in real-world coordinates.

        Returns:
            float or None: New steering angle (theta) in radians if a valid path is found.
                        None if no adjustment is necessary or possible.
        """
        steering_adjustment     = 0  # No adjustment initially
        boid_x_pos, boid_y_pos  = boid_pos

        # Calculate the boid's heading (theta) based on its velocity
        boid_theta_rad = math.atan2(boid_vel[1], boid_vel[0])

        # Compute the lookahead point and its map coordinates
        ahead_pos_cell, ahead_pos = self.calc_lookahead(boid_x_pos, boid_y_pos, boid_theta_rad)

        # Check if the initial path to the lookahead point is valid
        self.path_valid = self.check_path(boid_pos, ahead_pos)
        
        # If the path is valid, no adjustment is needed
        if self.path_valid:
            return None

        # Adjust steering direction incrementally until a valid path is found
        while not self.path_valid:
            steering_adjustment += self.step_angle

            # Check right adjustment
            new_theta_rad = boid_theta_rad + steering_adjustment
            ahead_pos_cell, ahead_pos = self.calc_lookahead(boid_x_pos, boid_y_pos, new_theta_rad)
            self.path_valid = self.check_path(boid_pos, ahead_pos)

            if self.path_valid:
                boid_theta_rad = new_theta_rad
                break

            # Ensure the new grid coordinates are within the map boundaries
            new_grid_x, new_grid_y = ahead_pos_cell
            if not self.map._in_map([new_grid_x, new_grid_y]):
                new_grid_x, new_grid_y = new_grid_x / 2, new_grid_y / 2

            # Check left adjustment
            new_theta_rad = boid_theta_rad - steering_adjustment
            ahead_pos_cell, ahead_pos = self.calc_lookahead(boid_x_pos, boid_y_pos, new_theta_rad)
            self.path_valid = self.check_path(boid_pos, ahead_pos)

            if self.path_valid:
                boid_theta_rad = new_theta_rad
                break

            # Ensure the new grid coordinates are within the map boundaries
            new_grid_x, new_grid_y = ahead_pos_cell
            if not self.map._in_map([new_grid_x, new_grid_y]):
                new_grid_x, new_grid_y = new_grid_x / 2, new_grid_y / 2

            # Stop if no adjustment is possible (shouldn't occur if step_angle is sensible)
            if steering_adjustment == 0.0:
                return None 

        # Return the adjusted boid heading (theta)
        return boid_theta_rad

    
    def _create_steering_force(self, steering_angle_rad):
        """
        Generate a steering force vector based on the given steering angle.

        Parameters:
            steering_angle (float): The angle in radians representing the steering direction.

        Returns:
            numpy.ndarray: A 2D vector [fx, fy] representing the steering force in the x and y directions.
        """
        # Calculate the x and y components of the steering force using the angle
        fx = math.cos(steering_angle_rad)  # x-component (cosine of the angle)
        fy = math.sin(steering_angle_rad)  # y-component (sine of the angle)

        # Return the steering force as a numpy array
        return np.array([fx, fy])

    
    def _steer_to_avoid(self, boid_pose, boid_vel, boid_goal=None):
        """
        Calculate a steering force to help the boid avoid obstacles while moving towards its goal.

        Parameters:
            boid_pose (list or tuple): Current pose of the boid [x, y, theta] in real-world coordinates.
            boid_vel (list or tuple): Current velocity of the boid [vx, vy].
            boid_goal (list or tuple, optional): Goal position [x, y]. Default is None.

        Returns:
            numpy.ndarray: A 2D steering force [fx, fy] to avoid obstacles.
        """
        # Compute the magnitude of the velocity
        vel_abs = np.linalg.norm(boid_vel)

        # If the boid is stationary, no steering force is needed
        if boid_vel == [0.0, 0.0]:
            return np.array([0.0, 0.0])

        # Determine the steering angle to avoid obstacles
        steering_angle = self._get_steering_direction(boid_pose[:2], boid_vel)

        # If no steering adjustment is required, return zero force
        if steering_angle is None:
            return np.array([0.0, 0.0])

        # Create the steering force based on the calculated steering angle
        self.steering_force = self._create_steering_force(steering_angle)

        # Return the steering force vector
        return self.steering_force 

