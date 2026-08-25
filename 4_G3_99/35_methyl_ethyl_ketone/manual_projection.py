import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag


class Projection(object):
    """
    This class is used to specify the manual projection matrix
    for CMA. It is stored as an object and is only needed when
    self.options.man_proj = True.
    """

    def __init__(self, options):

        self.options = options

    def run(self):

        unc = np.eye(1)

        ch3_str = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        ch2_str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)
        
        cc3_ang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)
        
        ch3_ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1],
            [0, 0, 0, 0, 1,-1]
        ]).T)
        
        ch2_ang = normalize(np.array([
            [4,-1,-1,-1,-1],
            [0, 1, 1,-1,-1],
            [0, 1,-1, 1,-1],
            [0, 1,-1,-1, 1]
        ]).T)

        cc_rot = normalize(np.array([
            [1, 1]
        ]).T)

        ch3_rot1 = normalize(np.array([
            [1, 1, 1]
        ]).T)

        ch3_rot2 = normalize(np.array([
            [1, 1, 1, 1, 1, 1]
        ]).T)
        
        Proj = block_diag(unc, unc, unc, unc, ch3_str, ch2_str, ch3_str, unc, cc3_ang, ch3_ang, ch2_ang, ch3_ang, cc_rot, ch3_rot1, ch3_rot2, unc)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15, 16, 18, 20, 22, 24, 25, 27],
            [6, 8, 11, 17, 19, 21, 23, 26, 28, 29, 30, 31, 32],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)