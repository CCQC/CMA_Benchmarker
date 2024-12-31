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
        str_c2c4 = np.eye(1)

        # 1
        str_c4c6 = np.eye(1)

        # 2
        str_c6c8 = np.eye(1)

        # 3
        str_c4h7 = np.eye(1)

        # 4-5
        str_ch2_c8 = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        # 6-8
        str_ch3_c2 = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 9
        ang_c2c4c6 = np.eye(1)

        # 10
        ang_ch_c4 = normalize(np.array([
            [1, -1]
        ]).T)

        # 11-12
        ang_ch2_c8 = normalize(np.array([
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 13-17
        ang_ch3_c2 = normalize(np.array([
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 2, -1, -1],
            [0, 0, 0, 0, 1, -1]
        ]).T)

        # 18
        rot_ch2_c8 = normalize(np.array([
            [1, 1]
        ]).T)

        # 19
        rot_ch3_c2 = normalize(np.array([
            [1, 1, 1]
        ]).T)

        # 20
        oop_c8 = np.eye(1)

        # 21
        oop_c4 = np.eye(1)

        # 22
        linx_c4 = np.eye(1)

        # 23
        liny_c4 = np.eye(1)

        Proj = block_diag(str_c2c4, str_c4c6, str_c6c8, str_c4h7, str_ch2_c8, str_ch3_c2, ang_c2c4c6,
                          ang_ch_c4, ang_ch2_c8, ang_ch3_c2, rot_ch2_c8, rot_ch3_c2, oop_c8, oop_c4, linx_c4, liny_c4)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 20, 22],
            [5, 8, 12, 15, 17, 18, 19, 21, 23]
        ], dtype=object)


def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
