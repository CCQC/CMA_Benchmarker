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

        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        cc3_ang_1 = normalize(np.array([
            [1, -1]
        ]).T)

        ch_ang = normalize(np.array([
            [1, -1]
        ]).T)

        cyc_6tor = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [1, 0, -1, 1, 0, -1],
            [-1, 2, -1, -1, 2, -1],
        ]).T)

        oh_rot = normalize(np.array([
            [1, 1]
        ]).T)

        Proj = block_diag(cc_6str, unc, unc, unc, unc, unc, unc, unc, cyc_6ang, cc3_ang_1, unc,
                          ch_ang, ch_ang, ch_ang, ch_ang, ch_ang, cyc_6tor, oh_rot, unc, unc, unc, unc, unc, unc)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
