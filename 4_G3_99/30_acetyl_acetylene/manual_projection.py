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
    
        unc = np.eye(1)
        
        ch3_str = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1]
        ]).T)
        
        cc3_ang_2 = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1]
        ]).T)
        
        ch3_ang = normalize(np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1]
        ]).T)
        
        tor = normalize(np.array([
        [1, 1, 1]
        ]).T)
        
        lin180 = normalize(np.array([
        [1,-1]
        ]).T)
        
        Proj = block_diag(unc,unc,unc,unc,unc,ch3_str,cc3_ang_2,ch3_ang,tor,unc,lin180,lin180,lin180,lin180)
        
        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
