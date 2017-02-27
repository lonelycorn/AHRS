import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

class VisualizationCamera:
    """
    An ideal pin-hole camera.
    Its intrisincs is described by an identity matrix.
    """
    def __init__(self, R_camera_to_world, t_camera_to_world):
        """
        :param R_camera_to_world orientation of the camera in world frame;
            an SO3.
        :param t_camera_to_world position of the camera in world frame;
            a 3-by-1 numpy array
        """
        R = R_camera_to_world.inverse().get_matrix()
        t = -np.matmul(R,  t_camera_to_world)

        # a camera projection defines the transformation from world ref
        # frame to camera ref frame
        self._projection_matrix = np.zeros((3, 4), dtype=np.float)
        for row in range(0, 3):
            for col in range(0, 3):
                self._projection_matrix[row][col] = R[row][col]
            self._projection_matrix[row][3] = t[row]

    def project_points(self, points_in_world):
        """
        :param points_in_world a list of 3-by-1 numpy arrays, which are
            the coordinates of the points in world ref frame.
        :return a list of 2-by-1 numpy arrays, which are the 2D image
            coordinates of the visable points.
        """
        result = []
        for p in points_in_world:
            p_in_world = np.array([p[0], p[1], p[2], 1.0], dtype=np.float)
            p_in_camera = np.matmul(self._projection_matrix, p_in_world)

            z = p_in_camera[2]
            if (z > 0): # in front of the camera
                x = p_in_camera[0] / z
                y = p_in_camera[1] / z
                result.append(np.array([x, y]))
            else:
                raise RuntimeError("point {} is invisible.".format(p_in_camera))
        return result
