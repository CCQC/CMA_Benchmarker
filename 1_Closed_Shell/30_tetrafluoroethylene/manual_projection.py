import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag

class Projection(object):
    """
    This class is used to specify the manual projection matrix
    for CMA. It is stored as an object and is only needed when
    self.options.man_proj = True.
    """

    def __init__(self,  options):

        self.options = options

    def run(self):

        CC_mat = np.eye(1)
        
        stretch_mat = 0.5 * np.array([
             [1, 1, 1, 1],
             [1, 1, -1, -1],
             [1, -1, 1, -1],
             [1, -1, -1, 1]
        ])
        
        # angles_mat = np.array([
            # [0.5,0.5,0.5,0.5],
            # [0.5,-0.5,-0.5,0.5],
            # [0.5,0.5,-0.5,-0.5],
            # [0.5,-0.5,0.5,-0.5]
        # ])
        angles_mat = np.array([
            [2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1],
            [2,-1,-1,-2, 1, 1],
            [0, 1,-1, 0,-1, 1],
        ]).T
        
        oop_mat = np.array([
            [1/np.sqrt(2),1/np.sqrt(2)],
            [1/np.sqrt(2),-1/np.sqrt(2)]
        ])
        
        tor_mat = np.array([
            [1/np.sqrt(2)],
            [1/np.sqrt(2)]
        ])
        
        Proj = block_diag(CC_mat,stretch_mat,angles_mat,tor_mat,oop_mat)
        
        
        Proj = 1/norm(Proj,axis=0)*Proj
        self.Proj = Proj                     
        self.sym_sort = np.array([
            [0,1,5],
            [11],
            [],
            [4,8],
            [9],
            [3,6],
            [2,7],
            [10],
            ],dtype=object)
