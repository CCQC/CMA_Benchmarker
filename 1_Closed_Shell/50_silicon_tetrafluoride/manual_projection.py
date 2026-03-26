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

        str1 = normalize(np.array([
            [ 1, 1, 1, 1],
            [-1,-1, 1, 1],
            # [1/np.sqrt(2),-1/np.sqrt(2), 0, 0],
            # [0, 0, 1/np.sqrt(2),-1/np.sqrt(2)]
            [-1, 1,-1, 1],
            [ 1,-1,-1, 1]
        ]).T)

        ang = normalize(np.array([
            [2, 2,-1,-1,-1,-1],
            [0, 0, 1,-1,-1, 1],
            [-1,1, 0, 0, 0, 0],
            [0, 0,-1, 0, 0, 1],
            [0, 0, 0, 1,-1, 0],
        ]).T)

        # ang1 = 1/np.sqrt(2) * np.array([
            # [1, 1],
            # [1,-1]
        # ])
        
        # ang2 = 0.5 * np.array([
            # [ 1, 1, 1],
            # [-1, 1,-1],
            # [ 1,-1,-1],
            # [-1,-1, 1],
        # ])
        
        Proj = block_diag(str1, ang)
        # Proj = block_diag(str1, ang1, ang2)
        
        
        
        self.Proj = Proj                     
        
        self.sym_sort = np.array([
            [0],
            [],
            [4,5],
            [],
            [1,2,3,6,7,8],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

