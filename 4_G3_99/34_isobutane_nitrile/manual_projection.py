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
        
        cc_2str = normalize(np.array([
        [1, 1],
        [1,-1]
        ]).T)
        
        ch3_2str = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1]
        ]).T)
        
        cc3_ang = normalize(np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1]
        ]).T)
        
        cc3h_ang = normalize(np.array([
        [2,-1,-1],
        [0, 1,-1]
        ]).T)
        
        ch3_2ang = normalize(np.array([
        [1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1],
        [1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
        [2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0],
        [2,-1,-1, 0, 0, 0,-2, 1, 1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1],
        [0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1],
        [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1],
        [0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1]
        ]).T)
        
        cc_2tor = normalize(np.array([
        [1, 1],
        [1,-1]
        ]).T)
        
        ch3_2rot = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1,-1,-1,-1]
        ]).T)
        
        oop = normalize(np.array([
        [1, 1, 1]
        ]).T)
        
        Proj = block_diag(unc,cc3_ang,unc,ch3_2str,cc3h_ang,cc3h_ang,ch3_2ang,ch3_2rot,oop,unc,unc)

        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
