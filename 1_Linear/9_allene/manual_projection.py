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
       
        CH_str = normalize(np.array([
        [1, 1, 1, 1],
        [1, 1,-1,-1],
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)
       
        CH_ang = normalize(np.array([
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1],
        ]).T)

        tor = normalize(np.array([
        [1, 1, 1, 1],
        ]).T)

        oop = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)

        lin = normalize(np.array([
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)

        Proj = block_diag(HA_str,CH_str,CH_ang,tor,oop,lin)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,2,6],
            [],
            [10],
            [1,3,7],
            [4,5,8,9,11,12,13,14],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

