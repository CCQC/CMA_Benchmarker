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

        mix_mat = 1/np.sqrt(2) * np.array([
            [1, 1],
            [1,-1]
        ])
        angles = np.array([
        [2.0/np.sqrt(6), -1.0/np.sqrt(6), -1.0/np.sqrt(6)],
        [0.0           ,  1.0/np.sqrt(2), -1.0/np.sqrt(2)],
        ])
        angles = angles.T
        #F-C combos, C-O, degen irreps 
        Proj = block_diag(mix_mat,np.eye(1),angles,np.eye(1))
        self.Proj = Proj                     
        self.sym_sort = np.array([
            [0,2,3],
            [],
            [1,4],
            [5],
            ],dtype=object)
