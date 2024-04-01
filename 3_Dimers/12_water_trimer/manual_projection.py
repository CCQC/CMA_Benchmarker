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
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T)
        HA_ang = normalize(np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T)
        tor = normalize(np.array([
        [ 1,-1, 1,-1, 1,-1],
        [ 1, 0,-1, 1, 0,-1],
        [-1, 2,-1,-1, 2,-1],
        ]).T) 

        # Three_anti_sym_str = np.array([
        # [1, 1, 1, 1, 1, 1],
        # [2, 2,-1,-1,-1,-1],
        # [0, 0, 1, 1,-1,-1],
        # [1,-1, 1,-1, 1,-1],
        # [2,-2,-1, 1,-1, 1],
        # [0, 0, 1,-1,-1, 1],
        # ]).T
        Anti_sym_str = np.array([
        [1, 1],
        [1,-1],
        ]).T

        Three_str = np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        Unc = np.array([
        [1],
        ]).T
        Twist = np.array([
        [1, 1],
        ]).T
        Three_anti_sym = np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1],
        ]).T
        Three_twist = np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 2,-1,-1,-1,-1],
        [0, 0, 1, 1,-1,-1],
        ]).T

        Proj = block_diag(HA_str,Unc,Unc,Unc,Anti_sym_str,Anti_sym_str,Anti_sym_str,HA_ang,tor)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

