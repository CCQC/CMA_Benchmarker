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
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T
        # 12-14, 37-38
        Unc = np.array([
        [1],
        ]).T
        # 15-17
        HA_ang = np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T
        # 18-23
        CH_ang = np.array([
        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [2,-2, 1,-1,-1, 1,-2, 2,-1, 1, 1,-1],
        [0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1],
        [2,-2,-1, 1,-1, 1, 2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1, 0, 0, 1,-1,-1, 1],
        ]).T
        # 24-25,35-36
        Anti_bend = np.array([
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        ]).T
        # 26-28
        tor = np.array([
        [1,-1, 1,-1, 1,-1],
        [0, 1,-1, 0, 1,-1],
        [2,-1,-1, 2,-1,-1],
        ]).T 
        # interfrag_tor = np.array([
        # [1, 1],
        # ]).T
        # 30
        # Anti_tors = np.array([
        # [1, 1,-1,-1],
        # ]).T
        # 29-34
        oop = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T

        Proj = block_diag(HA_str,CH_str,Unc,Unc,Unc,HA_ang,CH_ang,Anti_bend,tor,oop,Anti_bend,Unc,Unc)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj

        # self.sym_sort = np.array([
            # [0,6,12,13,14,29], # a1
            # [18], # a2
            # [1,19], # b1
            # [7,15,26,30], # b2
            # [[2,3],[8,9],[20,21],[24,25],[31,32],[35,36],[37,38]], # e1
            # [[4,5],[10,11],[16,17],[22,23],[27,28],[33,34]], # e2
            # ],dtype=object)
        # Flat degen
        # self.sym_sort = np.array([
            # [0,6,12,13,14,29], # a1
            # [18], # a2
            # [1,19], # b1
            # [7,15,26,30], # b2
            # [2,3,8,9,20,21,24,25,31,32,35,36,37,38], # e1
            # [4,5,10,11,16,17,22,23,27,28,33,34], # e2
            # ],dtype=object)
        # Only one partner
        # self.sym_sort = np.array([
            # [0,6,12,13,14,29], # a1
            # [18], # a2
            # [1,19], # b1
            # [7,15,26,30], # b2
            # [2,8,20,24,31,35,37], # e1
            # [4,10,16,22,27,33], # e2
            # ],dtype=object)
        # Second partner
        # self.sym_sort = np.array([
            # [0,6,12,13,14,29], # a1
            # [18], # a2
            # [1,19], # b1
            # [7,15,26,30], # b2
            # [3,9,21,23,32,36,38], # e1
            # [5,11,17,23,28,34], # e2
            # ],dtype=object)


def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

