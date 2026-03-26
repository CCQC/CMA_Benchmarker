import numpy as np
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

        bond_mat = np.eye(2)
        
        stretch_mat = np.array([
             [1/np.sqrt(3),  0           ,  2/np.sqrt(6)],
             [1/np.sqrt(3),  1/np.sqrt(2), -1/np.sqrt(6)],
             [1/np.sqrt(3), -1/np.sqrt(2), -1/np.sqrt(6)]
        ])
        
        beta_mat = np.array([
            [ 1, 0           , 0           , 0           ],
            [ 0, 2/np.sqrt(6), 1/np.sqrt(3), 0           ],
            [ 0,-1/np.sqrt(6), 1/np.sqrt(3), 1/np.sqrt(2)],
            [ 0,-1/np.sqrt(6), 1/np.sqrt(3),-1/np.sqrt(2)]
        ])
        
        alpha_mat = np.array([
            [-1/np.sqrt(6), 1/np.sqrt(2)], 
            [-1/np.sqrt(6),-1/np.sqrt(2)],
            [ 2/np.sqrt(6), 0           ]
        ])
        
        tor_mat = np.eye(1)
        
        Proj = block_diag(bond_mat,stretch_mat,beta_mat,alpha_mat,tor_mat)
        
        
        self.Proj = Proj                     
        self.sym_sort = np.array([
            [0,1,2,4,5,6,7,9],
            [3,8,10,11],
            ],dtype=object)
