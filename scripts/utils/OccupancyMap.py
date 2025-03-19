import numpy as np
from scipy.ndimage import morphology
from matplotlib import pyplot as plt

class OccupancyMap:

    # Constructor
    def __init__(self):
        # map: 2D array of integers which categorizes world occupancy
        self.map        = None 
        self.map_dim    = None
        self.resolution = None        # map sampling resolution (size of a cell))                            
        self.origin     = None            # world position of cell (0, 0) in self.map                      
        self.there_is_map = False          # set method has been called                          
    
    def set(self, data, resolution, origin):     
        # Set occupancy map, its resolution and origin. 
        self.map        = self.dilate_obstacles(data, 4)   # dilate the obstacles slightly  
        self.map_dim    = self.map.shape
        self.resolution = resolution
        self.origin     = np.array(origin)
        # rospy.loginfo('map dimensions: %s', self.map_dim)

    def _pos_to_cell(self, pos):
        """
        Convert a position in real-world coordinates to map cell indices.

        Parameters:
            pos (list or tuple): Real-world position [x, y] to convert.

        Returns:
            list: Corresponding cell indices [mx, my] in the map grid.
        """
        # Calculate the x-coordinate in map cells
        mx = (pos[0] - self.origin[0]) / self.resolution

        # Calculate the y-coordinate in map cells
        my = (pos[1] - self.origin[1]) / self.resolution

        # Return the cell indices, rounded to the nearest integer
        return [int(round(mx)), int(round(my))]
    
    def _cell_to_pos(self, cell):
        """
        Convert a map cell index [mx, my] to the corresponding real-world position [x, y].

        Parameters:
            cell (list or tuple): Cell index [mx, my] in the map.

        Returns:
            list: Real-world position [x, y].
        """
        # Convert the x cell index to real-world coordinates
        x = cell[0] * self.resolution + self.origin[0]

        # Convert the y cell index to real-world coordinates
        y = cell[1] * self.resolution + self.origin[1]

        # Return the real-world position
        return [x, y]

    
    def _distance_pos_to_cell(self, distance):
        """
        Convert a distance in real-world units to the equivalent number of cells.
        
        Parameters:
            distance (float): Distance in real-world units.
        
        Returns:
            int: Equivalent number of cells in the map.
        """
        return int(round(distance / self.resolution))
 
    
    def _in_map(self, loc):
        '''
        loc: list of index [x,y]
        returns True if location is in map and false otherwise
        '''
        [mx,my] = loc
        if mx >= self.map.shape[0]-1 or my >= self.map.shape[1]-1 or mx < 0 or my < 0:
            return False 
        return True
    
    def dilate_obstacles(self, grid, dilation_length):
        obstacle_mask = (grid == 100)
        structuring_element = np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]], dtype=bool)
        # structuring_element = np.array([[0, 0, 0],
        #                                 [0, 1, 0],
        #                                 [0, 0, 0]], dtype=bool)
        dilated_mask = morphology.binary_dilation(obstacle_mask, iterations=dilation_length, structure=structuring_element)
        dilated_grid = np.zeros_like(grid)
        dilated_grid[dilated_mask] = 100  # Set occupied cells to 100 in the dilated grid
        return dilated_grid
    
    def crop_pos(self, center_position_pos, size_pos):
        """
        Crop a region of the map based on the center position and size in real-world units.

        Parameters:
            center_position_pos (list or tuple): The center position of the crop in real-world coordinates [x, y].
            size_pos (float): Half-size of the crop area (width and height assumed to be equal) in real-world units.

        Returns:
            numpy.ndarray: Cropped map region.
        """

        # Convert center position to map cell index
        center_position_cell = self._pos_to_cell(center_position_pos)

        # Convert size from real-world units to cells
        size_cell = self._distance_pos_to_cell(size_pos)

        pad_value = 100
        # Initialize the cropped map with the padding value
        cropped_map = np.full((2*size_cell, 2*size_cell), pad_value, dtype=self.map.dtype)

        # Calculate the crop boundaries in cell indices
        x_min = center_position_cell[0] - size_cell
        x_max = center_position_cell[0] + size_cell
        y_min = center_position_cell[1] - size_cell
        y_max = center_position_cell[1] + size_cell

        # Determine the valid range within the map dimensions
        valid_x_min = max(0, x_min)
        valid_x_max = min(self.map_dim[0], x_max)
        valid_y_min = max(0, y_min)
        valid_y_max = min(self.map_dim[1], y_max)

        # Determine the corresponding range in the cropped map
        crop_x_min = max(0, -x_min)
        crop_x_max = crop_x_min + (valid_x_max - valid_x_min)
        crop_y_min = max(0, -y_min)
        crop_y_max = crop_y_min + (valid_y_max - valid_y_min)

        # Insert the valid map region into the cropped map
        cropped_map[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = \
            self.map[valid_x_min:valid_x_max, valid_y_min:valid_y_max]

        return cropped_map, [2*size_cell, 2*size_cell], self.resolution, [-size_pos, -size_pos], True

    def update(self, map, map_dim, resolution, origin, there_is_map):
        self.map        = map
        self.map_dim    = map_dim
        self.resolution = resolution
        self.origin     = origin
        self.there_is_map = there_is_map

    def show_orin_map(self):
        plt.matshow(self.map)
        plt.show()

    def show_map(self, center_position_pos, size_pos):
        plt.matshow(self.crop_pos(center_position_pos, size_pos))
        plt.show()