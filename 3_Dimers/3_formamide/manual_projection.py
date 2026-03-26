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
        # 0-1, 2-3, 8-9, 11-12, 19-20,24-25,26-27,28-29
        Asym_Str1 = np.array([
        [1,  1],
        [1, -1],
        ]).T
        # 4-7
        Asym_Str2 = np.array([
        [1,  1,  1,  1],
        [1,  1, -1, -1],
        [1, -1,  1, -1],
        [1, -1, -1,  1],
        ]).T
        # 10,23
        Unc = np.array([
        [1],
        ]).T
        # 13-14
        Asym_Bend1 = np.array([
        [1, -1,  1, -1],
        [1, -1, -1,  1],
        ]).T
        # 15-18
        Asym_Bend2 = np.array([
        [2, -1, -1,  2, -1, -1],
        [2, -1, -1, -2,  1,  1],
        [0,  1, -1,  0,  1, -1],
        [0,  1, -1,  0, -1,  1],
        ]).T
        # 21-22
        Asym_Tors = np.array([
        [1,  1,  1,  1],
        [1,  1, -1, -1],
        ]).T

        Proj = block_diag(Asym_Str1,Asym_Str1,Asym_Str2,Asym_Str1,Unc,Asym_Str1,Asym_Bend1,Asym_Bend2,Asym_Str1,Asym_Tors,Unc,Asym_Str1,Asym_Str1,Asym_Str1)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,2,4,6,8,10,11,13,15,17,19], # a_g
            [22,25,27,29], # b_g
            [21,23,24,26,28], # a_u
            [1,3,5,7,9,12,14,16,18,20]  # b_u
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

