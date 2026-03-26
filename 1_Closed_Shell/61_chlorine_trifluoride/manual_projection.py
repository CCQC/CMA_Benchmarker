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
        
        Proj = block_diag(np.eye(1),mix_mat,mix_mat,np.eye(1))
        self.Proj = Proj                     
        self.sym_sort = np.array([
            [0,1,3],
            [],
            [5],
            [2,4],
            ],dtype=object)
