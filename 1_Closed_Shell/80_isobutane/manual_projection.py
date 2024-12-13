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
        # 0
        HC_str = np.eye(1)
        # 1-3
        HA_str = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T)
        # 4-12
        CH_str = normalize(np.array([
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

        # 13-17
        HC_ang = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T)

        # 18-32
        CH_ang = normalize(np.array([
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

        # 33-35
        tor = normalize(np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1],
        ]).T) 

        Proj = block_diag(HC_str,HA_str,CH_str,HC_ang,CH_ang,tor)


        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,4,7,13,18,21,27],
            [10,24,30,33],
            [2,3,5,6,8,9,11,12,14,15,16,17,19,20,22,23,25,26,28,29,31,32,34,35],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
