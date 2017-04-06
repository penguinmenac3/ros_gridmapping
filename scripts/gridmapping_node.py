#!/usr/bin/env python
import threading
import sys
import math
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from ros_graph_slam.msg import PoseNode, Path2D
from nav_msgs.msg import OccupancyGrid, MapMetaData
import sensor_msgs.point_cloud2 as pc2

import numpy as np

class Map(object):
    """ 
    The Map class stores an occupancy grid as a two dimensional
    numpy array. 
    
    Public instance variables:

        width      --  Number of columns in the occupancy grid.
        height     --  Number of rows in the occupancy grid.
        resolution --  Width of each grid square in meters. 
        origin_x   --  Position of the grid cell (0,0) in 
        origin_y   --    in the map coordinate system.
        grid       --  numpy array with height rows and width columns.
        
    
    Note that x increases with increasing column number and y increases
    with increasing row number. 
    """

    def __init__(self, origin_x=-10, origin_y=-10, resolution=.1, 
                 width=1000, height=1000):
        """ Construct an empty occupancy grid.
        
        Arguments: origin_x, 
                   origin_y  -- The position of grid cell (0,0) in the
                                map coordinate frame.
                   resolution-- width and height of the grid cells 
                                in meters.
                   width, 
                   height    -- The grid will have height rows and width
                                columns cells.  width is the size of
                                the x-dimension and height is the size
                                of the y-dimension.
                                
         The default arguments put (0,0) in the center of the grid. 
                                
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution
        self.width = width 
        self.height = height 
        self.grid = np.zeros((height, width))

    def to_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """
     
        grid_msg = OccupancyGrid()

        # Set up the header.
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        # .info is a nav_msgs/MapMetaData message. 
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        
        # Rotated maps are not supported... quaternion represents no
        # rotation. 
        grid_msg.info.origin = Pose(Point(self.origin_x, self.origin_y, 0),
                               Quaternion(0, 0, 0, 1))

        # Flatten the numpy array into a list of integers from 0-100.
        # This assumes that the grid entries are probalities in the
        # range 0-1. This code will need to be modified if the grid
        # entries are given a different interpretation (like
        # log-odds).
        flat_grid = self.grid.reshape((self.grid.size,)) * 100
        grid_msg.data = list(np.round(flat_grid))
        return grid_msg

    def set_cell(self, x, y, val):
        """ Set the value of a cell in the grid. 

        Arguments: 
            x, y  - This is a point in the map coordinate frame.
            val   - This is the value that should be assigned to the
                    grid cell that contains (x,y).
        """
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            print("Map to small.")
        self.grid[iy, ix] = min(1.0, self.grid[iy, ix] + val)

class GridMapping(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.scans = {}
        self.frames_since_remap = 0
        self.remap_distance = 10
        self.weight = 0.05
        self.min_dist = 0.5
        rospy.init_node("gridmapping_node")

        self._map_pub = rospy.Publisher('map', OccupancyGrid, latch=True)
        self._map_data_pub = rospy.Publisher('map_metadata', MapMetaData, latch=True)

        rospy.Subscriber('~/trajectory2d', Path2D, self.trajectory_callack)
        rospy.Subscriber("~/pose_node", PoseNode, self.pose_node_callback)

    def publish_map(self, gmap):
        """ Publish the map. """
        grid_msg = gmap.to_message()
        self._map_data_pub.publish(grid_msg.info)
        self._map_pub.publish(grid_msg)

    def transform_point(self, pose, point):
        theta = pose.theta
        c = math.cos(theta)
        s = math.sin(theta)

        t0 = c * point[0] - s * point[1]
        t1 = s * point[0] + c * point[1]

        return [t0 + pose.x, t1 + pose.y]

    def pose_node_callback(self, data):
        self.lock.acquire()
        tmp = pc2.read_points(data.scan)
        tmp2 = []
        for x in tmp:
            tmp2.append([x[0], x[1]])
        self.scans[str(data.id)] = tmp2
        self.lock.release()

    def trajectory_callack(self, data):
        self.lock.acquire()
        self.frames_since_remap += 1
        if self.frames_since_remap < self.remap_distance:
            self.lock.release()
            return
            
        # Fill gridmap
        gmap = Map()
        for x in self.scans:
            pose = data.poses[int(x)]
            for p in self.scans[x]:
                # Skip if scan point is to close to robot. It maybe only sees itself.
                if p[0] * p[0] + p[1] * p[1] < self.min_dist * self.min_dist:
                    continue
                transformed = self.transform_point(pose, p)
                # Add to gridmap
                gmap.set_cell(transformed[0], transformed[1], self.weight)

        self.publish_map(gmap)
        self.frames_since_remap = 0
        self.lock.release()

def main():
    """
        The main function.

        It parses the arguments and sets up the GridMapping
    """
    GridMapping()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    sys.exit(0)

if __name__ == "__main__":
    main()
