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
    """
    Shared data between the visualization context and the application context.
    """
    def __init__(self):
        self._lock = RLock()
        self._stopped = False
        self._time = 0.0
        self._true_orientation = SO3()
        self._estimated_orientation = SO3()

    def acquire_lock(self):
        self._lock.acquire()

    def release_lock(self):
        self._lock.release()

    @property
    def time(self):
        with self._lock:
            # POD's don't need the extra copy
            return self._time

    @time.setter
    def time(self, t):
        with self._lock:
            # POD's don't need the extra copy
            self._time = t * 1.0

    @property
    def true_orientation(self):
        with self._lock:
            result = copy.deepcopy(self._true_orientation)
            return result

    @true_orientation.setter
    def true_orientation(self, R):
        with self._lock:
            self._true_orientation = copy.deepcopy(R)

    def stop(self):
        with self._lock:
            self._stopped = True

    @property
    def stopped(self):
        with self._lock:
            # POD's don't need the extra copy
            return self._stopped

    @property
    def estimated_orientation(self):
        with self._lock:
            result = copy.deepcopy(self._estimated_orientation)
            return result

    @estimated_orientation.setter
    def estimated_orientation(selfl, R):
        with self._lock:
            self._estimated_orientation = copy.deepcopy(R)



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

    def _get_frame_axes(self, R_from_body_to_world, scale=1.0):
        """
        calculate the axes of the reference frame which is transformed through R
        """
        axis_x = R_from_body_to_world * self._axis_x * scale + self._offset
        axis_y = R_from_body_to_world * self._axis_y * scale + self._offset
        axis_z = R_from_body_to_world * self._axis_z * scale + self._offset
        origin = R_from_body_to_world * self._origin * scale + self._offset
        
        points = [axis_x, axis_y, axis_z, origin]
        points = self._camera.project_points(points)

        body_axis_x_1 = [points[3][0], points[0][0]]
        body_axis_x_2 = [points[3][1], points[0][1]]
        body_axis_y_1 = [points[3][0], points[1][0]]
        body_axis_y_2 = [points[3][1], points[1][1]]
        body_axis_z_1 = [points[3][0], points[2][0]]
        body_axis_z_2 = [points[3][1], points[2][1]]
        
        return (body_axis_x_1, body_axis_x_2,
                body_axis_y_1, body_axis_y_2,
                body_axis_z_1, body_axis_z_2)

    def draw(self):
        """
        Visualize shared data
        """
        body_frame_scale = 0.5
        body_axis_width = 10
        world_axis_width = 5

        plt.clf()

        # true body ref frame
        R_from_body_to_world = self._shared_data.true_orientation
        body_axes = self._get_frame_axes(R_from_body_to_world, body_frame_scale)
        plt.plot(body_axes[0], body_axes[1], 'r--', linewidth=body_axis_width)
        plt.plot(body_axes[2], body_axes[3], 'g--', linewidth=body_axis_width)
        plt.plot(body_axes[4], body_axes[5], 'b--', linewidth=body_axis_width)

        # estimated body ref frame
        R_from_body_to_world = self._shared_data.estimated_orientation
        body_axes = self._get_frame_axes(R_from_body_to_world, body_frame_scale)
        plt.plot(body_axes[0], body_axes[1], 'r-', linewidth=body_axis_width)
        plt.plot(body_axes[2], body_axes[3], 'g-', linewidth=body_axis_width)
        plt.plot(body_axes[4], body_axes[5], 'b-', linewidth=body_axis_width)

        # world ref frame
        R_from_body_to_world = SO3()
        body_axes = self._get_frame_axes(R_from_body_to_world)
        plt.plot(body_axes[0], body_axes[1], 'r--', linewidth=world_axis_width)
        plt.plot(body_axes[2], body_axes[3], 'g--', linewidth=world_axis_width)
        plt.plot(body_axes[4], body_axes[5], 'b--', linewidth=world_axis_width)

        # visualization 
        """
        axes = plt.plot(body_axis_x_1, body_axis_x_2, 'r-', \
                        body_axis_y_1, body_axis_y_2, 'g-', \
                        body_axis_z_1, body_axis_z_2, 'b-', \
                        world_axis_x_1, world_axis_x_2, 'r--', \
                        world_axis_y_1, world_axis_y_2, 'g--', \
                        world_axis_z_1, world_axis_z_2, 'b--')
        plt.setp(axes, 'linewidth', axis_width)
        """
        plt.ylim((-VIEW_RANGE, VIEW_RANGE))
        plt.xlim((-VIEW_RANGE, VIEW_RANGE))
        plt.axes().set_aspect('equal')

        # all the plots are overlayed, so long as draw() has not been caleed.
        plt.draw()
        plt.title("time = %.2f s" % (self._shared_data.time))
        plt.show(block=False)

if (__name__ == '__main__'):
    pass
