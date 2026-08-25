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

        unc = np.eye(1)

        cc_6str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [2, 1, -1, -2, -1, 1],
            [0, 1, 1, 0, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        ch_4str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        ch2_2ang = normalize(np.array([
            [4, -1, -1, -1, -1, 4, -1, -1, -1, -1],
            [4, -1, -1, -1, -1, -4, 1, 1, 1, 1],
            [0, 1, 1, -1, -1, 0, 1, 1, -1, -1],
            [0, 1, 1, -1, -1, 0, -1, -1, 1, 1],
            [0, 1, -1, 1, -1, 0, 1, -1, 1, -1],
            [0, 1, -1, 1, -1, 0, -1, 1, -1, 1],
            [0, 1, -1, -1, 1, 0, 1, -1, -1, 1],
            [0, 1, -1, -1, 1, 0, -1, 1, 1, -1]
        ]).T)

        ch_4ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1]
        ]).T)

        cyc_6tor = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [0, -1, 1, 0, -1, 1],
            [2, -1, -1, 2, -1, -1],
        ]).T)

        ch_4oop = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        Proj = block_diag(cc_6str, ch2_2str, ch_4str, cyc_6ang,
                          ch2_2ang, ch_4ang, cyc_6tor, ch_4oop)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 4, 6, 10, 15, 17, 25],
            [5, 13, 16, 22, 28],
            [9, 20, 29, 34],
            [23, 33],
            [24, 31, 32],
            [8, 19, 30, 35],
            [1, 2, 12, 21, 27],
            [3, 7, 11, 14, 18, 26],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
