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
        # 0-5
        HA_str = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T
        # 6-11
        CH_str = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        # [2, 1,-1,-2,-1, 1],
        [-2,-1, 1, 2, 1,-1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T
        # 12-13, 37-38
        Unc = np.array([
        [1],
        ]).T
        # 14-16
        ThreeStr = np.array([
        [1, 1, 1],
        # [2,-1,-1],
        [-2, 1, 1],
        [0, 1,-1],
        ]).T
        # 17-19
        HA_ang = np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T
        # 20-25
        CH_ang = np.array([
        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [2,-2, 1,-1,-1, 1,-2, 2,-1, 1, 1,-1],
        [0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1],
        # [2,-2,-1, 1,-1, 1, 2,-2,-1, 1,-1, 1],
        [-2, 2, 1,-1, 1,-1,-2, 2, 1,-1, 1,-1],
        [0, 0, 1,-1,-1, 1, 0, 0, 1,-1,-1, 1],
        ]).T
        # 26-27,37-38
        Anti_bend = np.array([
        # [2,-1,-1],
        [-2, 1, 1],
        [0, 1,-1],
        ]).T
        # 28-30
        tor = np.array([
        [1,-1, 1,-1, 1,-1],
        [0, 1,-1, 0, 1,-1],
        # [2,-1,-1, 2,-1,-1],
        [-2, 1, 1,-2, 1, 1],
        ]).T 
        # interfrag_tor = np.array([
        # [1, 1],
        # ]).T
        # 30
        # Anti_tors = np.array([
        # [1, 1,-1,-1],
        # ]).T
        # 31-36
        oop = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        # [2, 1,-1,-2,-1, 1],
        [-2,-1, 1, 2, 1,-1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T

        Proj = block_diag(HA_str,CH_str,Unc,Unc,ThreeStr,HA_ang,CH_ang,Anti_bend,tor,oop,Unc,Unc)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj

        self.sym_sort = np.array([
            [0,6,12,13,14,31], # a1
            [20], # a2
            [1,21], # b1
            [7,17,28,32], # b2
            [[2,3],[9,8],[16,15],[22,23],[27,26],[34,33],[38,37]], # e1, 1 2, then 6 polarization
            [[4,5],[10,11],[18,19],[25,24],[29,30],[35,36]], # e2, sym, then antisym
            ],dtype=object)
        # Flat degen
        # self.sym_sort = np.array([
            # [0,6,12,13,14,31], # a1
            # [20], # a2
            # [1,21], # b1
            # [7,17,28,32], # b2
            # # [2,3,8,9,15,16,22,23,26,27,33,34,37,38], # e1
            # # [4,5,10,11,18,19,24,25,29,30,35,36], # e2
            # [2,9,16,22,27,34,38,
                # 3,8,15,23,26,33,37], # e1
            # [4,10,18,25,29,35,
                # 5,11,19,24,30,36], # e2
            # ],dtype=object)


def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

