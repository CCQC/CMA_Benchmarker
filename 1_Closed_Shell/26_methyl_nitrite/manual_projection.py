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

        HA_stretch = np.eye(3)
        
        stretch = np.array([
        1/np.sqrt(3)*np.array([ 1, 1, 1]),
        1/np.sqrt(6)*np.array([ 2,-1,-1]),
        1/np.sqrt(2)*np.array([ 0, 1,-1])
        ]).T

        HA_angles = np.eye(2)

        angles = np.array([
        1/np.sqrt(6)*np.array([ 1, 1, 1,-1,-1,-1]),
        1/np.sqrt(6)*np.array([ 2,-1,-1, 0, 0, 0]),
        1/np.sqrt(2)*np.array([ 0, 1,-1, 0, 0, 0]),
        1/np.sqrt(6)*np.array([ 0, 0, 0, 2,-1,-1]),
        1/np.sqrt(2)*np.array([ 0, 0, 0, 0, 1,-1]),
        ]).T

        tor = np.array([
        1/np.sqrt(1)*np.array([ 1, 0, 0, 0]),
        1/np.sqrt(3)*np.array([ 0, 1, 1, 1]),
        ]).T

        #raise RuntimeError

        Proj = block_diag(HA_stretch,stretch,HA_angles,angles,tor)
        self.sym_sort = np.array([
            [0,1,2,3,4,6,7,8,9,11],
            [5,10,12,13,14],
            ],dtype=object)


        self.Proj = Proj

