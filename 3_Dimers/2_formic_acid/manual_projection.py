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
        # 0-1, 2-3, 4-5, 6-7, 9-10, 11-12, 13-14, 15-16, 17-18, 20-21, 22-23
        Asym_Str = np.array([
        [1,  1],
        [1, -1],
        ]).T
        # 8, 19
        Unc = np.array([
        [1],
        ]).T
        # 13-14
        Asym_Bend = np.array([
        [1, -1,  1, -1],
        [1, -1, -1,  1],
        ]).T

        Proj = block_diag(Asym_Str,Asym_Str,Asym_Str,Asym_Str,Unc,Asym_Str,Asym_Str,Asym_Bend,Asym_Str,Asym_Str,Unc,Asym_Str,Asym_Str)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        
        self.sym_sort = np.array([
            [0,2,4,6,8,9,11,13,15], # a_g
            [18,21,23], # b_g
            [17,19,20,22], # a_u
            [1,3,5,7,10,12,14,16]  # b_u
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

