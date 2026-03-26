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
        """
        Example matrix block
        mat = normalize(np.array([
        [1,..,0],
        ]).T)
        """
       
        HA_str = np.eye(2)

        CH_str1 = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T) 

        CH_str2 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T) 

        HA_ang = np.eye(1)

        # 8-11
        CH_ang1 = normalize(np.array([
        [ 1, 1,-1,-1, 0],
        [ 1,-1, 1,-1, 0],
        [ 1,-1,-1, 1, 0],
        [-1,-1,-1,-1, 4],
        ]).T)
        # CH_ang1 = normalize(np.array([
        # [ 5, 0, 0, 0, 0, 1],
        # [ 0, 1, 1,-1,-1, 0],
        # [ 0, 1,-1, 1,-1, 0],
        # [ 0, 1,-1,-1, 1, 0],
        # [ 1, 0, 0, 0, 0, 5],
        # ]).T)
        # 12-16
        CH_ang2 = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T)

        tor1 = normalize(np.array([
        [1, 1, 1],
        ]).T)

        # Proj = block_diag(HA_str,CH_str1,CH_str2,CH_ang1,CH_ang2,tor1)
        Proj = block_diag(HA_str,CH_str1,CH_str2,HA_ang,CH_ang1,CH_ang2,tor1)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,3,5,7,9,11,12,13,15],
            [4,6,8,10,14,16,17],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
