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

        HA_str1 = np.eye(3)

        HA_str2 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
       
        CH_str = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1],
        ]).T)
       
        HA_ang1 = np.eye(2)

        HA_ang2 = normalize(np.array([
        [1,-1],
        ]).T)

        HA_ang3 = normalize(np.array([
        [1,-1],
        ]).T)

        CH_ang1 = normalize(np.array([
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

        tor1 = np.eye(1)

        tor2 = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        ]).T) 

        oop = np.eye(2)

        Proj = block_diag(HA_str1,HA_str2,CH_str,HA_ang1,HA_ang2,HA_ang3,CH_ang1,tor1,tor2,oop)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,3,5,7,9,11,12,15,17,19,21,23,27,28,29],
            [4,6,8,10,13,14,16,18,20,22,24,25,26],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

