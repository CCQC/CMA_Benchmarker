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
	    [1, 1],
        [1,-1],
        ]).T)
       
        CH_str1 = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1],
        ]).T)
       
        CH_str2 = normalize(np.array([
	    [1, 1],
        [1,-1],
        ]).T)
       
        HA_ang = np.eye(1)
        # 11-20
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

        CH_ang2 = normalize(np.array([
        [4,-1,-1,-1,-1],
        [0, 1, 1,-1,-1],
        [0, 1,-1, 1,-1],
        [0, 1,-1,-1, 1],
        ]).T)

        tor = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        ]).T)

        Proj = block_diag(HA_str,CH_str1,CH_str2,HA_ang,CH_ang1,CH_ang2,tor)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,2,4,8,10,11,13,17,21],
            [7,16,20,24,25],
            [6,9,15,19,22,26],
            [1,3,5,12,14,18,23],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

