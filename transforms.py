from numba import njit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """Check if array is a valid transform.

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """
    # check shape
    
    # use_apple
    #[[-9.99999940e-01  2.70786125e-17 -2.70786125e-17  0.00000000e+00]
    # [ 0.00000000e+00 -7.07106769e-01  7.07106769e-01  0.00000000e+00]
    # [-0.00000000e+00  7.07106709e-01  7.07106829e-01 -0.00000000e+00]
    # [ 1.02499986e+00 -2.89913774e-01 -6.43467247e-01  1.00000000e+00]]

    # hw2_multiview
    # [[ 1.00000006 -0.         -0.          0.        ]
    #  [-0.         -0.38268347  0.92387947 -0.62361866]
    #  [ 0.         -0.92387953 -0.38268344  0.25831133]
    #  [ 0.         -0.         -0.          1.        ]]

    if t.shape != (4,4):
        return False

    # check all elements are real
    real_check = np.all(np.isreal(t))

    # calc intermediates
    rtr = np.matmul(t[:3, :3].T, t[:3, :3])
    rrt = np.matmul(t[:3, :3], t[:3, :3].T)

    # make rtr and rrt are identity
    inverse_check = np.isclose(np.eye(3), rtr, atol=tolerance).all() and np.isclose(np.eye(3), rrt, atol=tolerance).all()

    # check det
    det_check = np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=tolerance).all()

    # make sure last row is correct
    last_row_check = np.isclose(t[3, :3], np.zeros((1, 3)), atol=tolerance).all() and np.isclose(t[3, 3], 1.0, atol=tolerance).all()

    # print("real_check", real_check)
    # print("inverse_check", inverse_check)
    # print("det_check", det_check)
    # print("last_row_check", last_row_check)

    return real_check and inverse_check and det_check and last_row_check

def transform_concat(t1, t2):
    """[summary]

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    if not transform_is_valid(t1):
        raise ValueError('Invalid input transform t1')
    if not transform_is_valid(t2):
        raise ValueError('Invalid input transform t2')

    return np.matmul(t1, t2)

def transform_point3s(t, ps):
    """Transfrom 3D points from one space to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points ps')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]

def transform_inverse(t):
    """Find the inverse of the transfom.

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    return np.linalg.inv(t)

@njit(parallel=True)
def camera_to_image(intrinsics, camera_points):
    """Project points in camera space to the image plane.

    Args:
        intrinsics (numpy.array [3, 3]): Pinhole intrinsics.
        camera_points (numpy.array [n, 3]): n 3D points (x, y, z) in camera coordinates.

    Raises:
        ValueError: If intrinsics are not the correct shape.
        ValueError: If camera points are not the correct shape.

    Returns:
        numpy.array [n, 2]: n 2D projections of the input points on the image plane.
    """
    if intrinsics.shape != (3, 3):
        raise ValueError('Invalid input intrinsics')
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError('Invalid camera point')

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    # find u, v int coords
    image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
    for i in prange(camera_points.shape[0]):
        image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fu / camera_points[i, 2]) + u0))
        image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fv / camera_points[i, 2]) + v0))

    return image_coordinates

def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.

    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.

    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    point_count = 0
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                point_count += 1

    # back project for each u, v
    point_cloud = np.zeros((point_count, 3))
    point_count = 0
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                point_cloud[point_count] = np.array([
                    (u - u0) * depth_image[v, u] / fu,
                    (v - v0) * depth_image[v, u] / fv,
                    depth_image[v, u]])
                point_count += 1

    return point_cloud