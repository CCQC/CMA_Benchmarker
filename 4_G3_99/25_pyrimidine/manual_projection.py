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

        ch_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        ch_ang = normalize(np.array([
            [1, -1]
        ]).T)

        ch_2ang = normalize(np.array([
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        cyc_6tor = normalize(np.array([
            [1,-1, 1,-1, 1,-1],
            [2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1]
        ]).T)

        ch_2oop = normalize(np.array([
            [1, 1],
            [1, -1],
        ]).T)

        Proj = block_diag(cc_6str, unc, unc, ch_2str, cyc_6ang,
                          ch_ang, ch_ang, ch_2ang, cyc_6tor, unc, unc, ch_2oop)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 3, 4, 6, 7, 8, 10, 11, 15],
            [18, 23],
            [1, 2, 5, 9, 12, 13, 14, 16],
            [17, 19, 20, 21, 22],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
