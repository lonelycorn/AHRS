import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
import numpy as np
from threading import RLock # re-entrant lock
import copy

from base.config import *
from base.SO3 import rodrigues, SO3
from visualization.visualization_camera import VisualizationCamera

class PlotterSharedData:
    
    def __init__(self):
        self._lock = RLock()
        self._stopped = False
        self._true_orientation = SO3()
        self._estimated_orientation = SO3()

    def acquire_lock(self):
        self._lock.acquire()

    def release_lock(self):
        self._lock.release()

    def get_true_orientation(self):
        with self._lock:
            result = copy.deepcopy(self._true_orientation)
            return result

    def set_true_orientation(self, R):
        with self._lock:
            self._true_orientation = copy.deepcopy(R)

    def stop(self):
        with self._lock:
            self._stopped = True

    @property
    def stopped(self):
        with self._lock:
            return self._stopped


class Plotter:
    """
    A collection of visualization utilities.
    """
    def __init__(self, shared_data, view_point=None):
        """
        :param shared_data: of class PlotterSharedData
        """
        self._shared_data = shared_data

        if (view_point is None):
            view_point = VIEW_POINT

        translation = np.array(view_point, dtype=np.float)

        if (np.linalg.norm(translation) < TOLERANCE):
            raise ValueError("Invalid view point: {}".format(view_point))

        z_camera_in_camera = np.array([0, 0, 1], dtype=np.float)
        z_camera_in_world = -np.array(view_point, dtype=np.float)

        # axis of rotation, from z_camera_in_camera to z_camera_in_world
        axis = np.cross(z_camera_in_camera, z_camera_in_world)

        # rotation angle, from z_camera_in_camera to z_camera_in_world
        sin_theta = np.linalg.norm(axis)
        cos_theta = np.dot(z_camera_in_world, z_camera_in_camera)
        theta = np.arctan2(sin_theta, cos_theta)

        if (sin_theta < TOLERANCE): # special case: two axes are colinear
            if (theta < 1.0): # view angle is 0
                R = np.eye(3)
            else: # view angle is pi
                R = -np.eye(3)
        else:
            # use rodrigues' formula to obtain the rotation matrix
            axis_unit = axis / np.linalg.norm(axis)
            R = rodrigues(axis_unit, theta)

        rotation = SO3(R)
        #rotation = SO3()
        self._camera = VisualizationCamera(rotation, translation)

        # generate 3 axes
        self._offset = np.array([-1, -1, 0], dtype=np.float)
        self._origin = np.array([0, 0, 0], dtype=np.float)
        self._axis_x = np.array([1, 0, 0], dtype=np.float)
        self._axis_y = np.array([0, 1, 0], dtype=np.float)
        self._axis_z = np.array([0, 0, 1], dtype=np.float)
    
    @property
    def stopped(self):
        return self._shared_data.stopped

    def draw(self):
        """
        Visualize shared data
        """
        R_from_body_to_world = self._shared_data.get_true_orientation()

        body_frame_scale = 0.5
        axis_width = 5

        # calculate the new body ref frame in world
        axis_x = R_from_body_to_world * self._axis_x * body_frame_scale + self._offset
        axis_y = R_from_body_to_world * self._axis_y * body_frame_scale + self._offset
        axis_z = R_from_body_to_world * self._axis_z * body_frame_scale + self._offset
        origin = R_from_body_to_world * self._origin * body_frame_scale + self._offset

        points = [axis_x, axis_y, axis_z, origin]
        points = self._camera.project_points(points)

        body_axis_x_1 = [points[3][0], points[0][0]]
        body_axis_x_2 = [points[3][1], points[0][1]]
        body_axis_y_1 = [points[3][0], points[1][0]]
        body_axis_y_2 = [points[3][1], points[1][1]]
        body_axis_z_1 = [points[3][0], points[2][0]]
        body_axis_z_2 = [points[3][1], points[2][1]]


        # world ref frame
        axis_x = self._axis_x + self._offset
        axis_y = self._axis_y + self._offset
        axis_z = self._axis_z + self._offset
        origin = self._origin + self._offset
        points = [axis_x, axis_y, axis_z, origin]
        points = self._camera.project_points(points)

        world_axis_x_1 = [points[3][0], points[0][0]]
        world_axis_x_2 = [points[3][1], points[0][1]]
        world_axis_y_1 = [points[3][0], points[1][0]]
        world_axis_y_2 = [points[3][1], points[1][1]]
        world_axis_z_1 = [points[3][0], points[2][0]]
        world_axis_z_2 = [points[3][1], points[2][1]]

        # visualization 
        plt.clf()
        axes = plt.plot(body_axis_x_1, body_axis_x_2, 'r-', \
                        body_axis_y_1, body_axis_y_2, 'g-', \
                        body_axis_z_1, body_axis_z_2, 'b-', \
                        world_axis_x_1, world_axis_x_2, 'r--', \
                        world_axis_y_1, world_axis_y_2, 'g--', \
                        world_axis_z_1, world_axis_z_2, 'b--')

        plt.setp(axes, 'linewidth', axis_width)
        plt.grid(True)
        plt.ylim((-VIEW_RANGE, VIEW_RANGE))
        plt.xlim((-VIEW_RANGE, VIEW_RANGE))
        plt.axes().set_aspect('equal')
        plt.draw()
        plt.show(block=False)

if (__name__ == '__main__'):
    pass
