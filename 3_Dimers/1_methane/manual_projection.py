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
        Sym_Stretches_1 = np.array([
        [1,  1],
        [1, -1],
        ]).T
        # 2-7
        Sym_Stretches_2 = np.array([
        [1,  1,  1,  1,  1,  1],
        [2, -1, -1,  2, -1, -1],
        [0,  1, -1,  0,  1, -1],
        [1,  1,  1, -1, -1, -1],
        [2, -1, -1, -2,  1,  1],
        [0,  1, -1,  0, -1,  1],
        ]).T
        # Sym_Stretches = np.array([
        # [1,  1,  1,  1,  1,  1,  1,  1],
        # [1,  1, -1, -1,  1,  1, -1, -1],
        # [1, -1,  1, -1,  1, -1,  1, -1],
        # [1, -1, -1,  1,  1, -1, -1,  1],
        # [1,  1,  1,  1, -1, -1, -1, -1],
        # [1,  1, -1, -1, -1, -1,  1,  1],
        # [1, -1,  1, -1, -1,  1, -1,  1],
        # [1, -1, -1,  1, -1,  1,  1, -1],
        # ]).T
        # 8
        Uncoupled_Coord = np.array([
        [1],
        ]).T
        # 9-18
        Bends = np.array([
        [1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1],
        [2, -1, -1,  0,  0,  0,  2, -1, -1,  0,  0,  0],
        [0,  1, -1,  0,  0,  0,  0,  1, -1,  0,  0,  0],
        [0,  0,  0,  2, -1, -1,  0,  0,  0,  2, -1, -1],
        [0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1],
        [1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [2, -1, -1,  0,  0,  0, -2,  1,  1,  0,  0,  0],
        [0,  1, -1,  0,  0,  0,  0, -1,  1,  0,  0,  0],
        [0,  0,  0,  2, -1, -1,  0,  0,  0, -2,  1,  1],
        [0,  0,  0,  0,  1, -1,  0,  0,  0,  0, -1,  1],
        ]).T
        # Bends = np.array([
        # [2,  2, -1, -1, -1, -1,  2,  2, -1, -1, -1, -1],
        # [0,  0,  1, -1, -1,  1,  0,  0,  1, -1, -1,  1],
        # [1, -1,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0],
        # [0,  0,  1,  0,  0, -1,  0,  0,  1,  0,  0, -1],
        # [0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1,  0],
        # [2,  2, -1, -1, -1, -1, -2, -2,  1,  1,  1,  1],
        # [0,  0,  1, -1, -1,  1,  0,  0, -1,  1,  1, -1],
        # [1, -1,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0],
        # [0,  0,  1,  0,  0, -1,  0,  0, -1,  0,  0,  1],
        # [0,  0,  0,  1, -1,  0,  0,  0,  0, -1,  1,  0],
        # ]).T
        # 20-23
        Anti_sym = np.array([
        [2, -1, -1,  2, -1, -1],
        [0,  1, -1,  0,  1, -1],
        [2, -1, -1, -2,  1,  1],
        [0,  1, -1,  0, -1,  1],
        ]).T
        # 19
        Twist = np.array([
        [1, 1, 1],
        ]).T
        
        # 22-23
        # Anti_sym = np.array([
        # [1,  1],
        # [1, -1],
        # ]).T

        Proj = block_diag(Sym_Stretches_1,Sym_Stretches_2,Uncoupled_Coord,Bends,Twist,Anti_sym)
        Proj = 1/norm(Proj,axis=0)*Proj
        
        self.Proj = Proj
        self.sym_sort = np.array([
            [0,2,8,9], # a_1g
            [], # a_2g
            [[3,4],[10,11],[12,13],[19,20]],  # e_g
            [23], # a_1u
            [1,5,14], # a_2u
            [[6,7],[15,16],[17,18],[21,22]],  # e_u
            ],dtype=object)
        # self.sym_sort = np.array([
            # [0,2,8,9], # a_1g
            # [], # a_2g
            # [3,4,10,11,12,13,20,21],  # e_g
            # [19], # a_1u
            # [1,5,14], # a_2u
            # [6,7,15,16,17,18,22,23],  # e_u
            # ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

