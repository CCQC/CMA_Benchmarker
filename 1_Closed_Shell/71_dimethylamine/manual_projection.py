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
        [1, 0, 0],
        [0, 1, 1],
        [0, 1,-1],
        ]).T)
       
        CH_str = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1],
        ]).T)
       
        HA_ang = normalize(np.array([
        # [2,-1,-1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 1,-1],
        ]).T)

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

        tor = normalize(np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1],
        ]).T)

        # oop = np.eye(1)*2/np.sqrt(6)
        # oop = np.eye(1)

        # Proj = block_diag(HA_str,CH_str,HA_ang,CH_ang,tor,oop)
        Proj = block_diag(HA_str,CH_str,HA_ang,CH_ang,tor)
        
        # Proj[10][23] = -1/np.sqrt(6)
        # Proj[11][23] = -1/np.sqrt(6)


        np.set_printoptions(linewidth=2000)
        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,3,5,7,9,10,12,14,16,18,20,23],
            [2,4,6,8,11,13,15,17,19,21,22],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

