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
        # 0-2,5-7
        Sym_3stretch = np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        # 3-4,8,11,17-18,20,22
        Unc = np.array([
        [1],
        ]).T
        # 9-10
        Sym_3bend = np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        # 12-16
        Methyl = np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T
        # 19,23
        Sym3 = np.array([
        [1, 1, 1],
        ]).T
        # 21
        Sym2 = np.array([
        [1, 1],
        ]).T

        Proj = block_diag(Sym_3stretch,Unc,Unc,Sym_3stretch,Unc,Sym_3bend,Unc,Methyl,Unc,Unc,Sym3,Unc,Sym2,Unc,Sym3)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        # self.sym_sort = np.array([
            # [0,1,3,4,5,6,8,9,11,12,13,15,17,18,23], # a'
            # [2,7,10,14,16,19,20,21,22], # a"
            # ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

