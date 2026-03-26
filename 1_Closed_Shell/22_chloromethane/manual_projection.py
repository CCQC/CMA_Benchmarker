import numpy as np
from scipy.linalg import block_diag
from numpy import linalg as LA


class Projection(object):
    """
    This class is used to specify the manual projection matrix
    for CMA. It is stored as an object and is only needed when
    self.options.man_proj = True.
    """

    def __init__(self,  options):

        self.options = options

    def run(self):

        stretches_mat = np.array([
        1/np.sqrt(1)*np.array([1, 0, 0, 0]),
        1/np.sqrt(3)*np.array([0, 1, 1, 1]),
        1/np.sqrt(6)*np.array([0, 2,-1,-1]),
        1/np.sqrt(2)*np.array([0, 0, 1,-1]),
        ]).T

        angles = np.array([
            [ 2.,  0, 1., 0,  0],
            [-1.,  1.,1., 0,  0],
            [-1., -1.,1., 0,  0],
            [ 0,   0,-1., 2., 0],
            [ 0,   0,-1.,-1., 1.],
            [ 0,   0,-1.,-1.,-1.]
        ])
        angles = np.transpose(angles)
        temp = angles.copy()
        for i in range(len(angles)):
            angles[i] = temp[i] / LA.norm(temp[i])
        angles = np.transpose(angles)
        
        Proj = block_diag(stretches_mat,angles)



        self.Proj = Proj

        self.sym_sort = np.array([
            [0,1,6],
            [],
            [2,3,4,5,7,8],
            ],dtype=object)
