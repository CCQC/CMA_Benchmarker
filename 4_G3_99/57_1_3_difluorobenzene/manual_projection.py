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

        cc_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        cc3_2ang = normalize(np.array([
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        ch_ang = normalize(np.array([
            [1, -1]
        ]).T)

        ch_2ang = normalize(np.array([
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        cyc_6tor = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [1, 0, -1, 1, 0, -1],
            [-1, 2, -1, -1, 2, -1],
        ]).T)

        ch2c_2oop = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        Proj = block_diag(cc_6str, cc_2str, unc, unc, ch_2str, cyc_6ang, cc3_2ang,
                          ch_ang, ch_ang, ch_2ang, cyc_6tor, ch2c_2oop, unc, unc, ch2c_2oop)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 3, 4, 6, 8, 9, 10, 12, 13, 15, 19],
            [23, 24, 28],
            [21, 22, 25, 26, 27, 29],
            [1, 2, 5, 7, 11, 14, 16, 17, 18, 20],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
