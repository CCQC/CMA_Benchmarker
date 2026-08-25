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
        
        # 26
        unc = np.eye(1)

        # 0-1, 2-3, 4-5
        cc_2str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        # 6-11, 12-17
        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 2,-1,-1],
            [2,-1,-1,-2, 1, 1],
            [0, 1,-1, 0, 1,-1],
            [0, 1,-1, 0,-1, 1]
        ]).T)

        # 18-19
        ch_2str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        # 20-25
        c4_2ang = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1],
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1,-2, 1, 1],
            [0, 1,-1, 0,-1, 1]
        ]).T)

        # 27-36, 37-46
        ch3_2ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1],
            [1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
            [2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0],
            [2,-1,-1, 0, 0, 0,-2, 1, 1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1],
            [0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1],
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1],
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1]
        ]).T)

        # 47-50
        hc4_2ang = normalize(np.array([
            [2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1],
            [2,-1,-1,-2, 1, 1],
            [0, 1,-1, 0,-1, 1]
        ]).T)

        # 51-52
        cc_2tor = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1,-1,-1]
        ]).T)

        # 53-54, 55-56
        ch3_2rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1]
        ]).T)

        Proj = block_diag(cc_2str, cc_2str, cc_2str, ch3_2str, ch3_2str, ch_2str, c4_2ang, unc, ch3_2ang, ch3_2ang, hc4_2ang, cc_2tor, ch3_2rot, ch3_2rot)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 26, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 48, 51, 53, 55],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 24, 25, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 49, 50, 52, 54, 56],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)