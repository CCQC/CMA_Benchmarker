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
        # 0-4
        # HA_str = normalize(np.array([
        # [2, 1, 1, 1, 1],
        # [0, 1, 1, -1,-1],
        # [4,-1,-1, -1, -1],
        # [0, 1,-1, 1,-1],
        # [0, 1,-1,-1, 1],
        # ]).T)

        # 1-4
        HA_str = normalize(np.array([
        [ 1, 1, 1, 1],
        [ 1, 1, -1, -1],
        [ 1,-1, 1,-1],
        [ 1,-1,-1, 1],
        ]).T)
       
        # 5-6
        CH_str1 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
       
        # 7-10
        CH_str2 = normalize(np.array([
        [1, 1, 1, 1],
        [1, 1,-1,-1],
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)
       
        # 11-12
        CH_ang1 = normalize(np.array([
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
        tor = np.eye(1)
        
        # 22-23
        oop = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)

        Proj = block_diag(np.eye(1),HA_str,CH_str1,CH_str2,CH_ang1,CH_ang2,tor,oop)
        # Proj = block_diag(HA_str,CH_str1,CH_str2,CH_ang1,CH_ang2,tor,oop)
        # Proj = block_diag(np.eye(1),HA_str,CH_str1,CH_str2,CH_ang1,CH_ang2,tor)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,5,7,9,13,15,21,22],
            [4,12,18,20],
            [2,6,17,19,23],
            [3,8,10,11,14,16],
            ],dtype=object)
        # self.sym_sort = np.array([
            # [0,1,5,7,9,11,15,17,23],
            # [4,14,20,22],
            # [2,6,12,19,21],
            # [3,8,10,13,16,18],
            # ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

