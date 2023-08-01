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
        
        CX_mat = np.eye(2)
        
        CH_mat = np.array([
             [1/np.sqrt(2), 1/np.sqrt(2)],
             [1/np.sqrt(2),-1/np.sqrt(2)]
        ])
        
        CX_ang_mat = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1],
        ]).T)
        
        # CH_ang_mat = np.array([
            # [ 2/np.sqrt(6), 0           ],
            # [-1/np.sqrt(6), 1/np.sqrt(2)],
            # [-1/np.sqrt(6),-1/np.sqrt(2)]
        # ])
        
        tor_mat = 1/np.sqrt(2) * np.array([
            [1],
            [1]
        ])
        
        oop_mat = np.eye(2)
        
        Proj = block_diag(CC_mat,CX_mat,CH_mat,CX_ang_mat,CX_ang_mat,tor_mat,oop_mat)
        # Proj = block_diag(CC_mat,CX_mat,CH_mat,CX_ang_mat,CH_ang_mat,tor_mat,oop_mat)
        
        
        self.Proj = Proj                     

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

