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
       
        HA_str = normalize(np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 2,-1,-1],
        [0, 0, 1,-1],
        ]).T)

        CH_str = normalize(np.array([
        [1, 1, 1, 1],
        [1, 1,-1,-1],
        [1,-1,-1, 1],
        [1,-1, 1,-1],
        ]).T) 

        NH_ang = normalize(np.array([
        [1,-1],
        ]).T) 

        CH_ang = normalize(np.array([
        [4,-1,-1,-1,-1, 4,-1,-1,-1,-1],
        [4,-1,-1,-1,-1,-4, 1, 1, 1, 1],
        [0, 1,-1, 1,-1, 0, 1,-1, 1,-1],
        [0, 1,-1, 1,-1, 0,-1, 1,-1, 1],
        [0, 1, 1,-1,-1, 0, 1, 1,-1,-1],
        [0, 1, 1,-1,-1, 0,-1,-1, 1, 1],
        [0, 1,-1,-1, 1, 0, 1,-1,-1, 1],
        [0, 1,-1,-1, 1, 0,-1, 1, 1,-1],
        ]).T)

        oop = np.eye(1)

        Proj = block_diag(HA_str,CH_str,NH_ang,CH_ang,oop)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,4,6,9,11,14,16,17],
            [3,5,7,8,10,12,13,15],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
