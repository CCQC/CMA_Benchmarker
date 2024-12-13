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
       
        HA_str = np.eye(1)

        CH_str = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T) 

        CH_ang = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T)

        tor = normalize(np.array([
        [1, 1, 1],
        ]).T)

        Proj = block_diag(HA_str,CH_str,CH_str,CH_ang,CH_ang,tor)

        self.Proj = Proj
        
        self.sym_sort = np.array([
            [0,1,4,7,12],
            [17],
            [2,3,5,6,8,9,10,11,13,14,15,16],
            ],dtype=object)


def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
