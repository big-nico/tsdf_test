# Stencil code based on that of Andy Zeng

import numpy as np
import open3d as o3d

from numba import njit, prange
from skimage import measure
from transforms import *

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, volume_bounds, voxel_size):
        """Initialize tsdf volume instance variables.

        Args:
            volume_bounds (numpy.array [3, 2]): rows index [x, y, z] and cols index [min_bound, max_bound].
                Note: units are in meters.
            voxel_size (float): The side length of each voxel in meters.

        Raises:
            ValueError: If volume bounds are not the correct shape.
            ValueError: If voxel size is not positive.
        """
        volume_bounds = np.asarray(volume_bounds)
        if volume_bounds.shape != (3, 2):
            raise ValueError('volume_bounds should be of shape (3, 2).')

        if voxel_size <= 0.0:
            raise ValueError('voxel size must be positive.')

        # Define voxel volume parameters
        self._volume_bounds = volume_bounds
        self._voxel_size = float(voxel_size)

        # truncation on SDF (max alowable distance away from a surface)
        self._truncation_margin = 2 * self._voxel_size

        # Adjust volume bounds and ensure C-order contiguous and calculate voxel bounds taking the voxel size into consideration
        self._voxel_bounds = np.ceil((self._volume_bounds[:,1]-self._volume_bounds[:,0])/self._voxel_size).copy(order='C').astype(int)
        self._volume_bounds[:,1] = self._volume_bounds[:,0]+self._voxel_bounds*self._voxel_size

        # volume min bound is the origin of the volume in world coordinates
        self._volume_origin = self._volume_bounds[:,0].copy(order='C').astype(np.float32)

        print('Voxel volume size: {} x {} x {} - # voxels: {:,}'.format(
            self._voxel_bounds[0],
            self._voxel_bounds[1],
            self._voxel_bounds[2],
            self._voxel_bounds[0]*self._voxel_bounds[1]*self._voxel_bounds[2]))

        # Initialize pointers to voxel volume in memory
        self._tsdf_volume = np.ones(self._voxel_bounds).astype(np.float32)

        # for computing the cumulative moving average of observations per voxel
        self._weight_volume = np.zeros(self._voxel_bounds).astype(np.float32)
        color_bounds = np.append(self._voxel_bounds, 3)
        self._color_volume = np.zeros(color_bounds).astype(np.float32) # rgb order

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
                range(self._voxel_bounds[0]),
                range(self._voxel_bounds[1]),
                range(self._voxel_bounds[2]),
                indexing='ij')

        self._voxel_coords = np.concatenate([
                xv.reshape(1,-1),
                yv.reshape(1,-1),
                zv.reshape(1,-1)], axis=0).astype(int).T


    @staticmethod
    @njit(parallel=True)
    def voxel_to_world(volume_origin, voxel_coords, voxel_size):
        """Convert from voxel coordinates to world coordinates
            (in effect scaling voxel_coords by voxel_size).

        Args:
            volume_origin (numpy.array [3, ]): The origin of the voxel
                grid in world coordinate space.
            voxel_coords (numpy.array [n, 3]): Each row gives the 3D coordinates of a voxel.
            voxel_size (float): The side length of each voxel in meters.

        Returns:
            numpy.array [n, 3]: World coordinate representation of each of the n 3D points.
        """
        volume_origin = volume_origin.astype(np.float32)
        voxel_coords = voxel_coords.astype(np.float32)
        world_points = np.empty_like(voxel_coords, dtype=np.float32)

        # NOTE: prange is used instead of range(...) to take advantage of parallelism.
        for i in prange(voxel_coords.shape[0]):
            for j in range(3):
                world_points[i, j] = volume_origin[j] + (voxel_size * voxel_coords[i, j])

        return world_points

    @staticmethod
    @njit(parallel=True)
    def integrate_volume_helper(tsdf_old, margin_distance, w_old, observation_weight):
        """Return updated tsdf and weight volumes by integrating information from the current observation.

        Args:
            tsdf_old (numpy.array [v, ]): v is equal to the number of voxels to be
                integrated at this timestep. Old tsdf values that need to be
                updated based on the current observation.
            margin_distance (numpy.array [v, ]): The tsdf trancation margin.
            w_old (numpy.array [v, ]): old weight values.
            observation_weight (float): Weight to give each new observation.

        Returns:
            numpy.array [v, ]: new tsdf values for entries in tsdf_old
            numpy.array [v, ]: new weights to be used in the future.
        """
        tsdf_new = np.empty_like(tsdf_old, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)

        for i in prange(len(tsdf_old)):
            w_new[i] = w_old[i] + observation_weight
            tsdf_new[i] = (w_old[i] * tsdf_old[i] + observation_weight * margin_distance[i]) / w_new[i]

        return tsdf_new, w_new

    def integrate(self, color_image, depth_image, camera_intrinsics, camera_pose, observation_weight=1.):
        """Integrate an RGB-D observation into the TSDF volume, by updating the weight volume,
            tsdf volume, and color volume.

        Args:
            color_image (numpy.array [h, w, 3]): An rgb image.
            depth_image ((numpy.array [h, w]): A z depth image.
            camera_intrensics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
            camera_pose (numpy.array [4, 4]): SE3 transform representing pose (camera to world)
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        """
        image_height, image_width = depth_image.shape
        color_image = color_image.astype(np.float32)

        # Convert voxel grid coordinates to world points
        world_points = self.voxel_to_world(self._volume_origin, self._voxel_coords, self._voxel_size)

        # Transform points in the volume to the camera coordinate system. Get depths and u, v projections
        camera_points = transform_point3s(transform_inverse(camera_pose), world_points)
        voxel_z = camera_points[:, 2]
        projection = camera_to_image(camera_intrinsics, camera_points)
        voxel_u, voxel_v = projection[:, 0], projection[:, 1]

        # Eliminate pixels not in the image bounds or that are behind the image plane
        valid_pixel = np.logical_and(voxel_u >= 0,
                                np.logical_and(voxel_u < image_width,
                                np.logical_and(voxel_v >= 0,
                                np.logical_and(voxel_v < image_height,
                                voxel_z > 0))))

        # Get depths for valid coordinates u, v from the depth image. Zero elsewhere.
        depths = np.zeros(voxel_u.shape)
        depths[valid_pixel] = depth_image[voxel_v[valid_pixel], voxel_u[valid_pixel]]

        # Calculate depth differences
        depth_diff = depths - voxel_z

        # Filter out zero depth values and cases where depth + truncation margin >= voxel_z
        valid_points = np.logical_and(depths > 0, depth_diff >= -self._truncation_margin)

        # Truncate and normalize
        margin_distance = np.maximum(-1, np.minimum(1, depth_diff / self._truncation_margin))
        valid_voxel_x = self._voxel_coords[valid_points, 0]
        valid_voxel_y = self._voxel_coords[valid_points, 1]
        valid_voxel_z = self._voxel_coords[valid_points, 2]

        # get weights and tsdf for the valid voxels, update based on the observation
        w_old = self._weight_volume[valid_voxel_x, valid_voxel_y, valid_voxel_z]
        tsdf_old = self._tsdf_volume[valid_voxel_x, valid_voxel_y, valid_voxel_z]
        valid_margin_distance = margin_distance[valid_points]
        tsdf_new, w_new = self.integrate_volume_helper(tsdf_old, valid_margin_distance, w_old, observation_weight)
        self._weight_volume[valid_voxel_x, valid_voxel_y, valid_voxel_z] = w_new
        self._tsdf_volume[valid_voxel_x, valid_voxel_y, valid_voxel_z] = tsdf_new

        # Integrate color
        color_old = self._color_volume[valid_voxel_x, valid_voxel_y, valid_voxel_z]
        r_old = color_old[:, 0]
        g_old = color_old[:, 1]
        b_old = color_old[:, 2]

        color_new = color_image[voxel_v[valid_points],voxel_u[valid_points]]
        r_new = color_new[:, 0]
        g_new = color_new[:, 1]
        b_new = color_new[:, 2]
        b_new = np.minimum(255., np.round((w_old*b_old + observation_weight*b_new) / w_new))
        g_new = np.minimum(255., np.round((w_old*g_old + observation_weight*g_new) / w_new))
        r_new = np.minimum(255., np.round((w_old*r_old + observation_weight*r_new) / w_new))

        self._color_volume[valid_voxel_x, valid_voxel_y, valid_voxel_z] = np.array([r_new, g_new, b_new]).T

    def get_volume(self):
        """Get the tsdf and color volumes.

        Returns:
            numpy.array [l, w, h]: l, w, h are the dimensions of the voxel grid in voxel space.
                Each entry contains the integrated tsdf value.
            numpy.array [l, w, h, 3]: l, w, h are the dimensions of the voxel grid in voxel space.
                3 is the channel number in the order r, g, then b.
        """
        return self._tsdf_volume, self._color_volume

    def get_mesh(self):
        """ Run marching cubes over the constructed tsdf volume to get a mesh representation.

        Returns:
            numpy.array [n, 3]: each row represents a 3D point.
            numpy.array [k, 3]: each row is a list of point indices used to render triangles.
            numpy.array [n, 3]: each row represents the normal vector for the corresponding 3D point.
            numpy.array [n, 3]: each row represents the color of the corresponding 3D point.
        """
        tsdf_volume, color_vol = self.get_volume()

        print("tsdf_volume shape: ", tsdf_volume.shape)

        # Marching cubes
        voxel_points, triangles, normals, _ = measure.marching_cubes(tsdf_volume, level=0)
        points_ind = np.round(voxel_points).astype(int)
        points = self.voxel_to_world(self._volume_origin, voxel_points, self._voxel_size)

        # Get vertex colors.
        rgb_vals = color_vol[points_ind[:,0], points_ind[:,1], points_ind[:,2]]
        colors_r = rgb_vals[:, 0]
        colors_g = rgb_vals[:, 1]
        colors_b = rgb_vals[:, 2]
        colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        colors = colors.astype(np.uint8)

        return points, triangles, normals, colors
