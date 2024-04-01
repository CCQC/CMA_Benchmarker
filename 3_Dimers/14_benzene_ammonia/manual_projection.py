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
        CH_ang = np.array([
        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [2,-2, 1,-1,-1, 1,-2, 2,-1, 1, 1,-1],
        [0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1],
        [2,-2,-1, 1,-1, 1, 2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1, 0, 0, 1,-1,-1, 1],
        ]).T
        tor = normalize(np.array([
        [ 1,-1, 1,-1, 1,-1],
        [ 1, 0,-1, 1, 0,-1],
        [-1, 2,-1,-1, 2,-1],
        ]).T) 
        oop = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T

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
        NH_ang = np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        Unc = np.array([
        [1],
        ]).T
        Anti_sym = np.array([
        [1,-1],
        ]).T
        Twist = np.array([
        [1, 1],
        ]).T
        NH_oop = np.array([
        [1, 1, 1],
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

        Proj = block_diag(HA_str,HA_str,Three_str,Unc,HA_ang,CH_ang,NH_ang,Anti_sym,Anti_sym,Unc,tor,Twist,Twist,oop,NH_oop)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

