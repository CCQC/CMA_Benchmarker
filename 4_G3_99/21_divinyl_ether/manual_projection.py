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

        cc_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        cc_2ang = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch_2ang = normalize(np.array([
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        ch2c_2ang = normalize(np.array([
            [2, -1, -1, 2, -1, -1],
            [2, -1, -1, -2, 1, 1],
            [0, 1, -1, 0, 1, -1],
            [0, 1, -1, 0, -1, 1]
        ]).T)

        cc_2tor = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch2_rot = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1]
        ]).T)

        ch_2out = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch2_2out = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        Proj = block_diag(cc_2str, cc_2str, ch_2str, ch2_2str, unc, cc_2ang,
                          ch_2ang, ch2c_2ang, cc_2tor, ch2_rot, ch_2out, ch2_2out)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 2, 4, 6, 8, 10, 11, 13, 15, 17],
            [19, 21, 23, 25],
            [20, 22, 24, 26],
            [1, 3, 5, 7, 9, 12, 14, 16, 18],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
