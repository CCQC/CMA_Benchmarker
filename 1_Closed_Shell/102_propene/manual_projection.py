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
        HA_str = np.eye(2)
        # 2-4
        CH_str1 = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T)
        # 5
        CH_str2 = np.eye(1)
        # 6-7
        CH_str3 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
        # 8-9
        HA_ang = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T)
        # 10-14
        CH_ang1 = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T)
        # 15-16
        CH_ang3 = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T)
        # 17
        tor1 = normalize(np.array([
        [1, 1, 1],
        ]).T)
        # 18
        tor2 = normalize(np.array([
        [1, 1],
        ]).T)
        # 19-20
        oop = np.eye(2)

        Proj = block_diag(HA_str,CH_str1,CH_str2,CH_str3,HA_ang,CH_ang1,CH_ang3,tor1,tor2,oop)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,3,5,6,7,8,9,10,11,13,15,16],
            [4,12,14,17,18,19,20],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

