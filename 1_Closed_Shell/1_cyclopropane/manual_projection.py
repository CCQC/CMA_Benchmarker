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
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T)
       
        # CH_str = normalize(np.array([
        # [1, 1, 1, 1, 1, 1],
        # [2, 2,-1,-1,-1,-1],
        # [0, 0, 1, 1,-1,-1],
        # [1,-1, 1,-1, 1,-1],
        # [2,-2,-1, 1,-1, 1],
        # [0, 0, 1,-1,-1, 1],
        # ]).T)

        # 3-8
        CH_str = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 2,-1,-1,-1,-1],
        [0, 0, 1, 1,-1,-1],
        [1,-1, 1,-1, 1,-1],
        [2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1],
        ]).T)
       
        # 9-20
        CH_ang = normalize(np.array([
        [4,-1,-1,-1,-1, 4,-1,-1,-1,-1, 4,-1,-1,-1,-1],
        [8,-2,-2,-2,-2,-4, 1, 1, 1, 1,-4, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 4,-1,-1,-1,-1,-4, 1, 1, 1, 1],
        [0, 1, 1,-1,-1, 0, 1, 1,-1,-1, 0, 1, 1,-1,-1],
        [0, 2, 2,-2,-2, 0,-1,-1, 1, 1, 0,-1,-1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 0,-1,-1, 1, 1],
        [0, 1,-1, 1,-1, 0, 1,-1, 1,-1, 0, 1,-1, 1,-1],
        [0, 2,-2, 2,-2, 0,-1, 1,-1, 1, 0,-1, 1,-1, 1],
        [0, 0, 0, 0, 0, 0, 1,-1, 1,-1, 0,-1, 1,-1, 1],
        [0, 1,-1,-1, 1, 0, 1,-1,-1, 1, 0, 1,-1,-1, 1],
        [0, 2,-2,-2, 2, 0,-1, 1, 1,-1, 0,-1, 1, 1,-1],
        [0, 0, 0, 0, 0, 0, 1,-1,-1, 1, 0,-1, 1, 1,-1],
        ]).T)

        Proj = block_diag(HA_str,CH_str,CH_ang)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0,3,9],
            [15],
            [1,2,4,5,10,11,16,17],
            [18],
            [6,12],
            [7,8,13,14,19,20],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

