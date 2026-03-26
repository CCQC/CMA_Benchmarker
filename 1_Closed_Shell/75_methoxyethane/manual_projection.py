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
        # 0-2
        HA_str = normalize(np.array([
        # [1, 0, 0],
        # [0, 1, 1],
        # [0, 1,-1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        ]).T)
       
        # 3-5
        CH_str1 = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T)
       
        # 6-7
        CH_str2 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
       
        # 8-10
        CH_str3 = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T)

        # 11-12
        HA_ang = np.eye(2)

        # 13-17
        CH_ang1 = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T)

        # 18-21
        CH_ang2 = normalize(np.array([
        [4,-1,-1,-1,-1],
        [0, 1, 1,-1,-1],
        [0, 1,-1, 1,-1],
        [0, 1,-1,-1, 1],
        ]).T)

        # 22-26
        CH_ang3 = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T)

        # 27
        tor1 = np.eye(1)

        # 28
        tor2 = normalize(np.array([
        [1, 1, 1],
        ]).T) 

        # 29
        tor3 = normalize(np.array([
        [1, 1, 1],
        ]).T) 

        Proj = block_diag(HA_str,CH_str1,CH_str2,CH_str3,HA_ang,CH_ang1,CH_ang2,CH_ang3,tor1,tor2,tor3)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,3,4,6,8,9,11,12,13,14,16,18,20,22,23,25],
            [5,7,10,15,17,19,21,24,26,27,28,29],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

