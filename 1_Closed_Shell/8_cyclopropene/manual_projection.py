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
        # 0-2
        HA_stretch = np.array([
        1/np.sqrt(3)*np.array([ 1, 1, 1]),
        1/np.sqrt(2)*np.array([ 1,-1, 0]),
        1/np.sqrt(6)*np.array([-1,-1, 2]),
        ]).T
        
        # 3-6
        stretch = np.array([
        1/np.sqrt(2)*np.array([ 1, 1, 0, 0]),
        1/np.sqrt(2)*np.array([ 1,-1, 0, 0]),
        1/np.sqrt(2)*np.array([ 0, 0, 1, 1]),
        1/np.sqrt(2)*np.array([ 0, 0, 1,-1])
        ]).T

        # 7-10
        angles = np.array([
        0.5/np.sqrt(5)*np.array([ 4,-1,-1,-1,-1]),
        1/np.sqrt(4)*np.array([ 0, 1, 1,-1,-1]),
        1/np.sqrt(4)*np.array([ 0, 1,-1, 1,-1]),
        1/np.sqrt(4)*np.array([ 0, 1,-1,-1, 1]),
        ]).T

        # 11-12
        angles2 = np.array([
        1/np.sqrt(4)*np.array([ 1,-1, 1,-1]),
        1/np.sqrt(4)*np.array([ 1,-1,-1, 1]),
        ]).T

        tor = np.array([
        1/np.sqrt(2)*np.array([ 1, 1]),
        ]).T
        # 13-14
        oop = np.array([
        # 1/np.sqrt(2)*np.array([ 1, 1]),
        1/np.sqrt(2)*np.array([ 1,-1]),
        ]).T

        #raise RuntimeError

        Proj = block_diag(HA_stretch,stretch,angles,angles2,np.eye(1),oop)
        # Proj = block_diag(HA_stretch,stretch,angles,angles2,tor,oop)


        self.Proj = Proj
        
        self.sym_sort = np.array([
            [0,2,3,5,7,11],
            [10,13],
            [4,8,14],
            [1,6,9,12],
            ],dtype=object)

