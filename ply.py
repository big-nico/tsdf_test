import numpy as np
import os

class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """
    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point.. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point.. Defaults to None.
        """
        super().__init__()

        # If ply path is None, load in triangles, point, normals, colors.
        # else load ply from file. If ply_path is specified AND other inputs
        # are specified as well, ignore other inputs.
        # If normals are not None make sure that there are equal number of points and normals.
        # If colors are not None make sure that there are equal number of colors and normals.

        self.triangles = []
        self.points = []
        self.normals = []
        self.colors = []

        if ply_path is not None:
            assert os.path.exists(ply_path)
            self.read(ply_path)
        else:
            # Set from input args.
            if triangles is not None:
                self.triangles = triangles

            if points is not None:
                self.points = points

            if normals is not None:
                self.normals = normals

            if colors is not None:
                self.colors = colors

        if len(self.normals) != 0:
            assert len(self.normals) == len(self.points)

        if len(self.colors) != 0:
            assert len(self.colors) == len(self.points)


    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        with open(ply_path, 'w') as f:
            # Write header.
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(len(self.points)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')

            if len(self.normals) != 0:
                f.write('property float nx\n')
                f.write('property float ny\n')
                f.write('property float nz\n')

            if len(self.colors) != 0:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')

            # Write faces header if dumping triangles.
            if len(self.triangles) != 0:
                f.write('element face {}\n'.format(len(self.triangles)))
                f.write('property list uchar int vertex_index\n')

            f.write('end_header\n')

            # Write points.
            for i in range(len(self.points)):
                f.write('{0} {1} {2}'.format(
                    self.points[i][0],
                    self.points[i][1],
                    self.points[i][2]))

                if len(self.normals) != 0:
                    f.write(' {0} {1} {2}'.format(
                        self.normals[i][0],
                        self.normals[i][1],
                        self.normals[i][2]))

                if len(self.colors) != 0:
                    f.write(' {0} {1} {2}'.format(
                        int(self.colors[i][0]),
                        int(self.colors[i][1]),
                        int(self.colors[i][2])))

                f.write('\n')

            # write triangles if they exist
            for triangle in self.triangles:
                f.write('3 {0} {1} {2}\n'.format(triangle[0], triangle[1], triangle[2]))

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        vertex_mode = False
        face_mode = False
        num_points = 0
        num_faces = 0
        index = 0

        self.points = []
        self.normals = []
        self.colors = []
        self.triangles = []

        parse_order = []

        with open(ply_path, 'r') as ps:
            for line in ps:
                line = line.split()

                if vertex_mode:
                    # Read in points and normals.
                    property_dict = {}

                    assert len(parse_order) == len(line)

                    for i, key in enumerate(parse_order):
                        property_dict[key] = float(line[i])

                    if ('x' in property_dict) and ('y' in property_dict) and ('z' in property_dict):
                        self.points.append([property_dict['x'], property_dict['y'], property_dict['z']])
                    if ('nx' in property_dict) and ('ny' in property_dict) and ('nz' in property_dict):
                        self.normals.append([property_dict['nx'], property_dict['ny'], property_dict['nz']])
                    if ('red' in property_dict) and ('green' in property_dict) and ('blue' in property_dict):
                        self.colors.append([property_dict['red'], property_dict['green'], property_dict['blue']])
                    index += 1
                    if index == num_points:
                        vertex_mode = False
                        face_mode = True
                        index = 0
                elif face_mode:
                    # Read in triangles.
                    self.triangles.append([int(i) for i in line[1:4]])
                    index += 1
                    if index == num_faces:
                        face_mode = False
                elif line[0] == 'element':
                    # set number of lines for vertices and faces.
                    if line[1] == 'vertex':
                        num_points = int(line[2])
                    elif line[1] == 'face':
                        num_faces = int(line[2])
                elif line[0] == 'property' and num_faces <= 0:
                    parse_order.append(line[2])
                elif line[0] == 'end_header':
                    vertex_mode = True
