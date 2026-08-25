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

        cc_3str = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1],
        ]).T)

        ch2_str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch3_str = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [2, -1, -1, -2, 1, 1],
            [0, 1, -1, 0, 1, -1],
            [0, 1, -1, 0, -1, 1]
        ]).T)

        ch3_ang = normalize(np.array([
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 2, -1, -1],
            [0, 0, 0, 0, 1, -1]
        ]).T)

        ch2c_ang = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        ch3_2ang = normalize(np.array([
            [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
            [1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1],
            [2, -1, -1, 0, 0, 0, 2, -1, -1, 0, 0, 0],
            [2, -1, -1, 0, 0, 0, -2, 1, 1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 2, -1, -1, 0, 0, 0, 2, -1, -1],
            [0, 0, 0, 2, -1, -1, 0, 0, 0, -2, 1, 1],
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1],
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 1]
        ]).T)

        ch2c_rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1]
        ]).T)

        ch3_rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).T)

        ch3_2rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        ]).T)

        Proj = block_diag(unc, cc_3str, ch2_str, ch3_str, ch3_2str, ch3_ang,
                          ch2c_ang, ch3_ang, ch3_2ang, ch2c_rot, ch3_rot, ch3_2rot)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 2, 4, 6, 7, 9, 11, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 30, 32, 34, 36, 41],
            [3, 5, 8, 10, 12, 14, 17, 19, 22, 25, 27, 29, 31, 33, 35, 37, 38, 39, 40],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
