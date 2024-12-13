import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import norm

class Projection(object):
    """
    This class is used to specify the manual projection matrix
    for CMA. It is stored as an object and is only needed when
    self.options.man_proj = True.
    """

    def __init__(self,  options):

        self.options = options

    def run(self):

        CC_mat = np.eye(1)
        
        CO_mat = (1/np.sqrt(2)) * np.array([
            [1,1],
            [1,-1]
        ])
        
        # angles_mat = np.array([
            # [0.5,0.5,0.5,0.5],
            # [0.5,-0.5,-0.5,0.5],
            # [0.5,0.5,-0.5,-0.5],
            # [0.5,-0.5,0.5,-0.5]
        # ])
        # 5-8
        angles_mat = normalize(np.array([
            [ 1,-1, 0, 1,-1, 0],
            [ 1,-1, 0,-1, 1, 0],
            [-1,-1, 2,-1,-1, 2],
            [-1,-1, 2, 1, 1,-2],
        ]).T)
        
        oop_mat = np.array([
            [1/np.sqrt(2),1/np.sqrt(2)],
            [1/np.sqrt(2),-1/np.sqrt(2)]
        ])
        
        # tor_mat = np.array([
            # [1/np.sqrt(2)],
            # [1/np.sqrt(2)]
        # ])
        
        # Proj = block_diag(CC_mat,CO_mat,CO_mat,angles_mat,tor_mat,oop_mat)
        Proj = block_diag(CC_mat,CO_mat,CO_mat,angles_mat,CC_mat,oop_mat)
        
        
        self.Proj = Proj                     
        self.sym_sort = np.array([
            [0,1,3,5,7],
            [11],
            [9,10],
            [2,4,6,8],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
