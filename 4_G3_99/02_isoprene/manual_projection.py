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

        # 0-2
        str_c4_c2 = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 3
        str_c3c4 = np.eye(1)

        # 4
        str_c3c8 = np.eye(1)

        # 5-6
        str_ch2_c1 = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        # 7-8
        str_ch2_c4 = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        # 9-11
        str_ch3_c5 = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 12
        ang_c2c3c4 = np.eye(1)

        # 13-14
        ang_c4_c2 = normalize(np.array([
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 15
        ang_ch_c3 = normalize(np.array([
            [1, -1]
        ]).T)

        # 16-17
        ang_ch2c_c4 = normalize(np.array([
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 18-19
        ang_ch2c_c1 = normalize(np.array([
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        # 20-24
        ang_ch3_c5 = normalize(np.array([
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 2, -1, -1],
            [0, 0, 0, 0, 1, -1]
        ]).T)

        # 25
        rot_c1c5_c2c3 = normalize(np.array([
            [1, 1]
        ]).T)

        # 26
        rot_ch2_c1 = normalize(np.array([
            [1, 1]
        ]).T)

        # 27
        rot_ch2_c4 = normalize(np.array([
            [1, 1]
        ]).T)

        # 28
        rot_ch3_c5 = normalize(np.array([
            [1, 1, 1]
        ]).T)

        # 29
        oop_c2 = normalize(np.array([
            [1, 1, 1]
        ]).T)

        # 30
        oop_c4 = np.eye(1)

        # 31
        oop_c1 = np.eye(1)

        # 32
        oop_c3 = np.eye(1)

        Proj = block_diag(str_c4_c2, str_c3c4, str_c3c8, str_ch2_c1, str_ch2_c4, str_ch3_c5, ang_c2c3c4, ang_c4_c2, ang_ch_c3,
                          ang_ch2c_c4, ang_ch2c_c1, ang_ch3_c5, rot_c1c5_c2c3, rot_ch2_c1, rot_ch2_c4, rot_ch3_c5, oop_c2, oop_c4, oop_c1, oop_c3)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 23],
            [11, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        ], dtype=object)


def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
