from image import read_rgb, read_depth
import numpy as np
import os
from ply import Ply
import time
import tsdf
from camera import *
from matplotlib import pyplot as plt
import open3d as o3d

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


    depth_points = []
    dimensions = depth_image.shape
    for v in range(dimensions[0]):
        for u in range(dimensions[1]):
            z = depth_image[v, u]
            if z > 0:
                X = (u - intrinsics[0, 2])/intrinsics[0,0] * z
                Y = (v - intrinsics[1, 2])/intrinsics[1,1] * z
                Z = z
                depth_point = [X, Y, Z]
                depth_points.append(depth_point)
    array = np.array(depth_points)
    return array

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

    # make sure it is a 4x4 matrix

    dimensions = t.shape
    if dimensions[0] != 4 or dimensions[1] != 4:
        return False
    
    # make sure the 3 x 3 matrix is valid

    R = t[:3, :3]

    r_flat = R.reshape(1, -1)
    r_flat = r_flat[0]

    # real number verification
    for element in r_flat:
        if isinstance(element, float) or isinstance(element, int):
            continue
        else:
            return False

    # last row match
    last_row = t[3]
    array = np.array([0,0,0,1])

    if not np.array_equiv(last_row, array):
        return False


    # transpose match
    subtraction = np.subtract(R.T@R, R@R.T)
    tolerance_matrix = np.full((3, 3), tolerance)
    if not np.all(abs(subtraction) <= tolerance_matrix):
        return False

    # determinant within tolerance of one
    det = np.linalg.det(R)
    if not (1 - tolerance <= det <= 1 + tolerance):
        return False

    # transform must be valid
    return True

def transform_point3s(t, ps):
    """Transform 3D points from one space to another.

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
    else:
        transformation_matrix = t[:3, :3]
        translation_matrix = t[:3,3]
        transformed_points = []
    
        for point in ps:
            transformed_point = transformation_matrix @ point
            transformed_point = transformed_point + translation_matrix
            transformed_points.append(list(transformed_point))

        arr = np.array(transformed_points)
        return arr

if __name__ == "__main__":
    # Set bounds based on max and min in each dimension in the world space.
    image_count = 16
    camera = Camera(
        image_size=(512, 512),
        near=0.01,
        far=10.0,
        fov_w=70.0,
    )
    camera_intrensics = np.loadtxt("use_apple/use_apple_instrinsics.txt", delimiter=' ')

    #load the pcd file
    pcd = o3d.io.read_point_cloud("use_apple/use_apple_gt_pointcloud.pcd")

    points = np.asarray(pcd.points)
    
    #visualize in matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()

    #get the volume bounds 
    min_x = np.min(points[:,0])
    max_x = np.max(points[:,0])
    min_y = np.min(points[:,1])
    max_y = np.max(points[:,1])
    min_z = np.min(points[:,2])
    max_z = np.max(points[:,2])

    volume_bounds = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

    print(volume_bounds)

    #find a good voxel size based on the volume bounds
    voxel_size = (max_x - min_x)/200

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_volume = tsdf.TSDFVolume(volume_bounds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together
    start_time = time.time()
    for i in range(image_count):
        print("Fusing frame %d/%d"%(i+1, image_count))

        # Read RGB-D image and camera pose
        color_image = read_rgb("use_apple/images/use_apple_image_{}.jpg".format(i))
        depth_image = read_depth("use_apple/depths/use_apple_depth_{}.png".format(i))
        camera_pose = cam_view2pose(np.load("use_apple/views/use_apple_view_matrix_{}.npy".format(i)))

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_volume.integrate(color_image, depth_image, camera_intrensics, camera_pose, observation_weight=1.)

    fps = image_count / (time.time() - start_time)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    points, faces, normals, colors = tsdf_volume.get_mesh()
    mesh = Ply(triangles=faces, points=points, normals=normals, colors=colors)
    mesh.write(os.path.join('supplemental', 'mesh.ply'))

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to point_cloud.ply...")
    pc = Ply(points=points, normals=normals, colors=colors)
    pc.write(os.path.join('supplemental', 'point_cloud.ply'))
