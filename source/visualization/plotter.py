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
        self._text = ""
        self._true_orientation = None
        self._estimated_orientation = None

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
    def text(self):
        with self._lock:
            # POD's don't need the extra copy
            return self._text

    @text.setter
    def text(self, t):
        with self._lock:
            # POD's don't need the extra copy
            self._text = str(t)

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
    def estimated_orientation(self, R):
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
        rotation = SO3.from_two_directions(z_camera_in_camera, z_camera_in_world)

        self._camera = VisualizationCamera(rotation, translation)

        # generate 3 axes
        self._offset = np.array([-0.5, -0.5, 0], dtype=np.float)
        self._origin = np.array([0, 0, 0], dtype=np.float)
        self._axis_x = np.array([1, 0, 0], dtype=np.float)
        self._axis_y = np.array([0, 1, 0], dtype=np.float)
        self._axis_z = np.array([0, 0, 1], dtype=np.float)
    
    @property
    def stopped(self):
        return self._shared_data.stopped

    def _get_frame_points(self, R_from_body_to_world, scale=1.0):
        """
        Calculate the canvas coordinate of the reference frame, which
        is characterized by R_from_body_to_world
        """
        # calculate new end points in world ref frame.
        axis_x = R_from_body_to_world * self._axis_x * scale + self._offset
        axis_y = R_from_body_to_world * self._axis_y * scale + self._offset
        axis_z = R_from_body_to_world * self._axis_z * scale + self._offset
        origin = R_from_body_to_world * self._origin * scale + self._offset
        
        points = [axis_x, axis_y, axis_z, origin]
        points = self._camera.project_points(points)

        # calculate canvas coordinates
        # NOTE: for the camera image, x-axis points to the right and
        #       y-axis points downward; for the canvas, x-axis points
        #       to the right and y-axis points up.
        body_axis_x_1 = [ points[3][0],  points[0][0]]
        body_axis_x_2 = [-points[3][1], -points[0][1]]
        body_axis_y_1 = [ points[3][0],  points[1][0]]
        body_axis_y_2 = [-points[3][1], -points[1][1]]
        body_axis_z_1 = [ points[3][0],  points[2][0]]
        body_axis_z_2 = [-points[3][1], -points[2][1]]
        
        points = (body_axis_x_1, body_axis_x_2,
                  body_axis_y_1, body_axis_y_2,
                  body_axis_z_1, body_axis_z_2)
        return points 

    def draw(self):
        """
        Visualize shared data
        """
        true_body_frame_scale = 0.75
        body_frame_scale = 0.5
        body_axis_width = 10
        world_axis_width = 5

        plt.clf()

        # true body ref frame
        R_from_body_to_world = self._shared_data.true_orientation
        if (R_from_body_to_world is not None):
            points = self._get_frame_points(R_from_body_to_world, true_body_frame_scale)
            plt.plot(points[0], points[1], 'r--', linewidth=body_axis_width)
            plt.plot(points[2], points[3], 'g--', linewidth=body_axis_width)
            plt.plot(points[4], points[5], 'b--', linewidth=body_axis_width)

        # estimated body ref frame
        R_from_body_to_world = self._shared_data.estimated_orientation
        if (R_from_body_to_world is not None):
            points = self._get_frame_points(R_from_body_to_world, body_frame_scale)
            plt.plot(points[0], points[1], 'r-', linewidth=body_axis_width)
            plt.plot(points[2], points[3], 'g-', linewidth=body_axis_width)
            plt.plot(points[4], points[5], 'b-', linewidth=body_axis_width)

        # world ref frame
        R_from_body_to_world = SO3()
        points = self._get_frame_points(R_from_body_to_world)
        plt.plot(points[0], points[1], 'r--', linewidth=world_axis_width)
        plt.plot(points[2], points[3], 'g--', linewidth=world_axis_width)
        plt.plot(points[4], points[5], 'b--', linewidth=world_axis_width)

        # axes
        plt.ylim((-VIEW_RANGE, VIEW_RANGE))
        plt.xlim((-VIEW_RANGE, VIEW_RANGE))
        plt.axes().set_aspect('equal')

        # all the plots are overlayed, so long as draw() has not been caleed.
        plt.title("time = %.2f s" % (self._shared_data.time))
        plt.text(-VIEW_RANGE, -VIEW_RANGE, self._shared_data.text, fontsize=20, color='r')
        plt.draw()
        plt.show(block=False)

if (__name__ == '__main__'):
    pass
