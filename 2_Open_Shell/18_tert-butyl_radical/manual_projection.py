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

        HA_str = normalize(np.array([
        [ 1, 1, 1],
        [ 2,-1,-1],
        [ 0, 1,-1],
        ]).T)
        
        # 3-11
        NH_str = normalize(np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2,-1,-1,-1,-1,-1,-1],
        [0, 0, 0, 1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1, 2,-1,-1],
        [4,-2,-2,-2, 1, 1,-2, 1, 1],
        [0, 0, 0, 2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1, 0, 1,-1],
        [0, 2,-2, 0,-1, 1, 0,-1, 1],
        [0, 0, 0, 0, 1,-1, 0,-1, 1],
        ]).T)

        # 12-13
        HA_ang = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T)

        # 14-28
        NH_ang = normalize(np.array([
        [1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1],
        [2, 2, 2,-2,-2,-2,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
        [2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0],
        [4,-2,-2, 0, 0, 0,-2, 1, 1, 0, 0, 0,-2, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0],
        [0, 2,-2, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1],
        [0, 0, 0, 4,-2,-2, 0, 0, 0,-2, 1, 1, 0, 0, 0,-2, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1],
        [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1],
        [0, 0, 0, 0, 2,-2, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1],
        ]).T)

        # 29-31
        tor = normalize(np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1],
        ]).T) 

        # 32
        oop = normalize(np.array([
        [1, 1, 1],
        ]).T) 

        Proj = block_diag(HA_str,NH_str,HA_ang,NH_ang,tor,oop)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,3,6,14,17,23,32],
            [9,20,26,29],
            [1,2,4,5,7,8,10,11,12,13,15,16,18,19,21,22,24,25,27,28,30,31],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

