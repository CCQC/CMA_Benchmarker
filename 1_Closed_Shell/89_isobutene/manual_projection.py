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
        [ 1, 0, 0],
        [ 0, 1, 1],
        [ 0, 1,-1],
        ]).T)
       
        # 3-4
        CH_str1 = normalize(np.array([
        [1, 1],
        [1,-1]
        ]).T)
       
        # 5-10
        CH_str2 = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1],
        ]).T)
        # CH_str2 = normalize(np.array([
        # [1, 1],
        # [1,-1]
        # ]).T)
       
        # CH_str3 = normalize(np.array([
        # [1, 1, 1, 1],
        # [1, 1,-1,-1],
        # [1,-1, 1,-1],
        # [1,-1,-1, 1],
        # ]).T)

        # 11-12
        HA_ang = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T)

        # 13-14
        CH_ang1 = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T)

        # 15-24
        CH_ang2 = normalize(np.array([
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
        # CH_ang2 = normalize(np.array([
        # [1, 1],
        # [1,-1]
        # ]).T)
       
        # CH_ang3 = normalize(np.array([
        # [1, 1, 1, 1],
        # [1, 1,-1,-1],
        # [1,-1, 1,-1],
        # [1,-1,-1, 1],
        # ]).T)

        
        # CH_ang4 = normalize(np.array([
        # [2,-1,-1, 2,-1,-1],
        # [2,-1,-1,-2, 1, 1],
        # [0, 1,-1, 0, 1,-1],
        # [0, 1,-1, 0,-1, 1],
        # ]).T)

        # 25
        tor1 = normalize(np.array([
        [1, 1],
        ]).T) 

        # 26-27
        tor2 = normalize(np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1],
        ]).T) 
        
        # 28-29
        oop = np.eye(2)
 
 
        Proj = block_diag(HA_str, CH_str1, CH_str2, HA_ang, CH_ang1, CH_ang2, tor1, tor2, oop)
        # Proj = block_diag(HA_str,CH_str1,CH_str2,CH_str3,HA_ang,CH_ang1,CH_ang2,CH_ang3,CH_ang4,tor1,tor2,oop)


        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,3,5,7,11,13,15,17,21],
            [10,20,24,25,26],
            [9,19,23,27,28,29],
            [2,4,6,8,12,14,16,18,22],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
