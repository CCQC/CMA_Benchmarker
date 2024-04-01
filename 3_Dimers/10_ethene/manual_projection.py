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
    
        Anti_sym = np.array([
        [1, 1],
        [1,-1],
        ]).T

        Eight_stretch = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1,-1,-1, 1, 1,-1,-1],
        [1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1],
        [1, 1, 1, 1,-1,-1,-1,-1],
        [1, 1,-1,-1,-1,-1, 1, 1],
        [1,-1, 1,-1,-1, 1,-1, 1],
        [1,-1,-1, 1,-1, 1, 1,-1],
        ]).T

        Unc = np.array([
        [1],
        ]).T
        
        CH2_bends = np.array([
        [2,-1,-1, 2,-1,-1, 2,-1,-1, 2,-1,-1],
        [2,-1,-1,-2, 1, 1, 2,-1,-1,-2, 1, 1],
        [0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1],
        [0, 1,-1, 0,-1, 1, 0, 1,-1, 0,-1, 1],
        [2,-1,-1, 2,-1,-1,-2, 1, 1,-2, 1, 1],
        [2,-1,-1,-2, 1, 1,-2, 1, 1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1,-0,-1, 1, 0,-1, 1],
        [0, 1,-1, 0,-1, 1,-0,-1, 1, 0, 1,-1],
        ]).T
        
        Anti_bend = np.array([
        [1,-1],
        ]).T

        Anti_tors1 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1,-1,-1,-1,-1],
        ]).T

        Sym_tors = np.array([
        [1, 1, 1, 1],
        ]).T

        Anti_tors2 = np.array([
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T
        
        Oop = np.array([
        [1, 1, 1, 1],
        [1, 1,-1,-1],
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T

        Proj = block_diag(Anti_sym,Eight_stretch,Unc,CH2_bends,Anti_bend,Anti_bend,Anti_tors1,Sym_tors,Anti_tors2,Oop)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

