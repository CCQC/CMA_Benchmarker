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

        stretches_mat = np.array([
        [(1/np.sqrt(3)), ( 1/np.sqrt(3)), (1/np.sqrt(3))],
        [(2/np.sqrt(6)), (-1/np.sqrt(6)), (-1/np.sqrt(6))],
        [0             , (1/np.sqrt(2)), (-1/np.sqrt(2))],
        ]).T
        
        angles = np.array([
        [(2/np.sqrt(6)), (-1/np.sqrt(6)), (-1/np.sqrt(6))],
        [0,              (1/np.sqrt(2)),  (-1/np.sqrt(2))],
        ])
        
        angles = angles.T
        
        oop_mat = np.array([
        [(1/np.sqrt(3)), (1/np.sqrt(3)), (1/np.sqrt(3))],
        ])
        oop_mat = oop_mat.T
        
        #raise RuntimeError
        
        Proj = block_diag(stretches_mat,angles,oop_mat)
        
        
        self.Proj = Proj                     
       
        self.sym_sort = np.array([
            [0,5],
            [],
            [1,2,3,4],
            ],dtype=object)
