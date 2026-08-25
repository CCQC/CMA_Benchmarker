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
        
        # 0
        unc = np.eye(1)

        # 1-4, 5-8
        cc_4str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1,-1,-1],
            [1,-1, 1,-1],
            [1,-1,-1, 1]
        ]).T)

        # 9-10
        cc_2str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        # 11-14, 15-18
        ch_4str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1,-1,-1],
            [1,-1, 1,-1],
            [1,-1,-1, 1]
        ]).T)

        # 19-24
        cyc6_2ang = normalize(np.array([
            [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
            [2,-1,-1, 2,-1,-1, 2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1],
            [1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1],
            [2,-1,-1, 2,-1,-1,-2, 1, 1,-2, 1, 1],
            [0, 1,-1, 0, 1,-1, 0,-1, 1, 0,-1, 1]
        ]).T)

        # 25-28, 29-32
        ch_4ang = normalize(np.array([
            [1,-1, 1,-1, 1,-1, 1,-1],
            [1,-1, 1,-1,-1, 1,-1, 1],
            [1,-1,-1, 1, 1,-1,-1, 1],
            [1,-1,-1, 1,-1, 1, 1,-1]
        ]).T)

        # 33-38
        cyc6_2tor = normalize(np.array([
            [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
            [2,-1,-1, 2,-1,-1, 2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1],
            [1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1],
            [2,-1,-1, 2,-1,-1,-2, 1, 1,-2, 1, 1],
            [0, 1,-1, 0, 1,-1, 0,-1, 1, 0,-1, 1]
        ]).T)

        # 39
        fus2_but = normalize(np.array([
            [1,-1]
        ]).T)

        # 40-43, 44-47
        ch_4oop = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1,-1,-1],
            [1,-1, 1,-1],
            [1,-1,-1, 1]
        ]).T)

        Proj = block_diag(unc, cc_4str, cc_4str, cc_2str, ch_4str, ch_4str, cyc6_2ang, ch_4ang, ch_4ang, cyc6_2tor, fus2_but, ch_4oop, ch_4oop)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 5, 9, 11, 15, 20, 25, 29],
            [36, 37, 40, 45],
            [35, 43, 46],
            [4, 8, 14, 18, 22, 24, 28, 32],
            [33, 34, 42, 47],
            [3, 7, 10, 13, 17, 23, 27, 31],
            [2, 6, 12, 16, 19, 21, 26, 30],
            [38, 39, 41, 44],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
