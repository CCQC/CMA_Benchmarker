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
        # 0-3
        HA_str = normalize(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1,-1],
        ]).T)
       
        # 4-5
        CH_str = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
       
        # 6-9
        CH_str2 = normalize(np.array([
        [1, 1, 1, 1],
        [1, 1,-1,-1],
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)
       
        # 10
        HA_ang = normalize(np.array([
        [1,-1, 1,-1],
        ]).T)

        # 11-12
        CH_ang = normalize(np.array([
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)
        # 13-20
        CH_ang2 = normalize(np.array([
        [4,-1,-1,-1,-1, 4,-1,-1,-1,-1],
        [4,-1,-1,-1,-1,-4, 1, 1, 1, 1],
        [0, 1, 1,-1,-1, 0, 1, 1,-1,-1],
        [0, 1, 1,-1,-1, 0,-1,-1, 1, 1],
        [0, 1,-1, 1,-1, 0, 1,-1, 1,-1],
        [0, 1,-1, 1,-1, 0,-1, 1,-1, 1],
        [0, 1,-1,-1, 1, 0, 1,-1,-1, 1],
        [0, 1,-1,-1, 1, 0,-1, 1, 1,-1],
        ]).T)

        # 21
        tor = normalize(np.array([
        [1,-1, 1,-1],
        ]).T)

        # 22-23
        oop = normalize(np.array([
        [1, 1],
        # [1,-1],
        ]).T)

        Proj = block_diag(HA_str,CH_str,CH_str2,HA_ang,CH_ang,CH_ang2,tor,np.eye(1),oop)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,4,6,11,13,17],
            [9,16,20,21,22],
            [8,15,19,23],
            [3,5,7,10,12,14,18],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

