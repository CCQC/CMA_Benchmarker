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
        
        # 6, 11, 29, 34, 35
        unc = np.eye(1)

        # 0-5
        cc_6str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [2, 1, -1, -2, -1, 1],
            [0, 1, 1, 0, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        # 7-10, 12-13
        ch_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        # 14-16
        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        # 17
        cc3_ang_1 = normalize(np.array([
            [1, -1]
        ]).T)

        # 18-21
        ch_2ang = normalize(np.array([
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        # 22
        ch_ang = normalize(np.array([
            [1, -1]
        ]).T)

        # 23-24
        ch2c_ang = normalize(np.array([
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 25-27
        cyc_6tor = normalize(np.array([
            [1,-1, 1,-1, 1,-1],
            [0,-1, 1, 0,-1, 1],
            [2,-1,-1, 2,-1,-1],
        ]).T)

        # 28
        ch2c_rot = normalize(np.array([
            [1, 1, 1, 1]
        ]).T)

        # 30-33
        ch_2oop = normalize(np.array([
            [1, 1],
            [1, -1],
        ]).T)

        Proj = block_diag(cc_6str, unc, ch_2str, ch_2str, unc, ch_2str, cyc_6ang, cc3_ang_1,
                          ch_2ang, ch_2ang, ch_ang, ch2c_ang, cyc_6tor, ch2c_rot, unc, ch_2oop, ch_2oop, unc, unc)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 3, 4, 6, 7, 9, 11, 12, 14, 15, 18, 20, 23, 25, 26, 29, 31, 33, 34, 35],
            [1, 2, 5, 8, 10, 13, 16, 17, 19, 21, 22, 24, 27, 28, 30, 32],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
