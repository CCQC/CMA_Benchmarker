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
        
        # 0-1
        HA_str = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
       
        # 2-7
        CH_str = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1],
        ]).T)
       
        # 8
        HA_ang = np.eye(1)

        # 9-18
        CH_ang = normalize(np.array([
        [1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1],
        [1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
        [2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0],
        [2,-1,-1, 0, 0, 0,-2, 1, 1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1],
        [0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1],
        [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1],
        [0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1],
        ]).T)

        # 19-20
        tor = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        ]).T)

        Proj = block_diag(HA_str,CH_str,HA_ang,CH_ang,tor)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,2,4,8,9,11,15],
            [6,13,17,19],
            [7,14,18,20],
            [1,3,5,10,12,16],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

