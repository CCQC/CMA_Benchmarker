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
        # 0-1
        Anti_sym = np.array([
        [1, 1],
        [1,-1],
        ]).T
        # 2-9
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
        # 10
        Unc = np.array([
        [1],
        ]).T
        # 11-18
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
        # 19-20,24-25
        Anti_tors2 = np.array([
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T
        # 
        # Anti_bend = np.array([
        # [1,-1],
        # ]).T
        # 21-22
        Anti_tors1 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1,-1,-1,-1,-1],
        ]).T
        # 23
        Sym_tors = np.array([
        [1, 1, 1, 1],
        ]).T
        # 26-29
        Oop = np.array([
        [1, 1, 1, 1],
        [1, 1,-1,-1],
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T

        Proj = block_diag(Anti_sym,Eight_stretch,Unc,CH2_bends,Anti_tors2,Anti_tors1,Sym_tors,Anti_tors2,Oop)
        # Proj = block_diag(Anti_sym,Eight_stretch,Unc,CH2_bends,Anti_bend,Anti_bend,Anti_tors1,Sym_tors,Anti_tors2,Oop)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        
        self.sym_sort = np.array([
            [0,2,4,10,11,13], # a_1
            [29], # a_2
            [23,28], # b_1
            [1,6,8,15,17], # b_2
            [[3,7],[5,9],[12,16],[14,18],[19,20],[21,22],[24,25],[26,27]]  # e
            ],dtype=object)
        # self.sym_sort = np.array([
            # [0,2,4,10,11,13], # a_1
            # [29], # a_2
            # [23,28], # b_1
            # [1,6,8,15,17], # b_2
            # # [3,7,5,9,12,16,14,18,19,20,21,22,24,25,26,27]  # e
            # [3,5,12,14,19,21,24,26,7,9,16,18,20,22,25,27]  # e
            # ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

