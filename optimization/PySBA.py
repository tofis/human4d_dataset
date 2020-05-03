"""

MIT License (MIT)

Copyright (c) FALL 2016, Jahdiel Alvarez

Author: Jahdiel Alvarez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import cv2

class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, pp, imgs, fs):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:

            cameraArray with shape (n_cameras, 6) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance.

            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.

            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

            point_ind with shape (n_observations,)
                    contatins indices of points (from 0 to n_points - 1) involved in each observation.

            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices
        self.pp = pp
        self.imgs = imgs
        self.fs = fs 

        points_proj = self.rotate(points3D[point2DIndices], cameraArray[cameraIndices][:, :3])
        points_proj += cameraArray[cameraIndices][:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]       

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, init_points, cameraArray, init_cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(init_points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # f = cameraArray[:, 6]
        # k1 = cameraArray[:, 7]
        # k2 = cameraArray[:, 8]
        # n = np.sum(points_proj ** 2, axis=1)
        # r = 1 + k1 * n + k2 * n ** 2
        # r = 1
        # points_proj *= (r * f)[:, np.newaxis]
        # points_proj *= f[:, np.newaxis]
        points_proj *= np.asarray(self.fs)[self.cameraIndices][:, np.newaxis]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], self.points3D[point_indices], camera_params[camera_indices], self.cameraArray[camera_indices])
        
        imgs_c = []
        for img in self.imgs:
            imgs_c.append(img.copy())

        p_index = 0
        for p in points_2d:            
            imgs_c[camera_indices[p_index]] = cv2.drawMarker(imgs_c[camera_indices[p_index]], 
                (int(p[0]) + self.pp[self.cameraIndices[p_index]][0], int(p[1]) + self.pp[self.cameraIndices[p_index]][1]), 
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=15,
                thickness=2)
            p_index += 1
        p_index = 0
        for p in points_proj:
            imgs_c[camera_indices[p_index]] = cv2.drawMarker(imgs_c[camera_indices[p_index]], 
                (int(p[0]) + self.pp[self.cameraIndices[p_index]][0], int(p[1]) + self.pp[self.cameraIndices[p_index]][1]), 
                (0, 255, 0),
                markerType=cv2.MARKER_SQUARE,
                markerSize=15,
                thickness=2)
            p_index += 1

        for i in range(len(imgs_c)):
            cv2.imshow(str(i), cv2.transpose(imgs_c[i]))
        cv2.waitKey(10)

        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numCameras * 6 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(6):
            A[2 * i, cameraIndices * 6 + s] = 1
            A[2 * i + 1, cameraIndices * 6 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 6 + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)
        self.init_params = A.copy()
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))

        params = self.optimizedParams(res.x, numCameras, numPoints)

        return params





