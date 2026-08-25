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
        unc = np.eye(1) # A1

        # 1-3
        cc_3str = normalize(np.array([
            [1, 1, 1],      # A1
            [2, -1, -1],    # E
            [0, 1, -1],     # E
        ]).T)

        # 4-12
        ch3_3str = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],        # A1
            [2, 2, 2, -1, -1, -1, -1, -1, -1],  # E
            [0, 0, 0, 1, 1, 1, -1, -1, -1],     # E
            [2, -1, -1, 2, -1, -1, 2, -1, -1],  # A1
            [4, -2, -2, -2, 1, 1, -2, 1, 1],    # E
            [0, 0, 0, 2, -1, -1, -2, 1, 1],     # E
            [0, 1, -1, 0, 1, -1, 0, 1, -1],     # A2
            [0, 2, -2, 0, -1, 1, 0, -1, 1],     # E
            [0, 0, 0, 0, 1, -1, 0, -1, 1]       # E
        ]).T)

        # 13-17
        clc4_ang = normalize(np.array([
            [1, 1, 1, -1, -1, -1],  # A1
            [2, -1, -1, 0, 0, 0],   # E
            [0, 1, -1, 0, 0, 0],    # E
            [0, 0, 0, 2, -1, -1],   # E
            [0, 0, 0, 0, 1, -1]     # E
        ]).T)

        # 18-32
        ch3_3ang = normalize(np.array([
            [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],    # A1
            [2, 2, 2, -2, -2, -2, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1],    # E
            [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1],       # E
            [2, -1, -1, 0, 0, 0, 2, -1, -1, 0, 0, 0, 2, -1, -1, 0, 0, 0],       # A1
            [4, -2, -2, 0, 0, 0, -2, 1, 1, 0, 0, 0, -2, 1, 1, 0, 0, 0],         # E
            [0, 0, 0, 0, 0, 0, 2, -1, -1, 0, 0, 0, -2, 1, 1, 0, 0, 0],          # E
            [0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0],          # A2
            [0, 2, -2, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0],          # E
            [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0],           # E
            [0, 0, 0, 2, -1, -1, 0, 0, 0, 2, -1, -1, 0, 0, 0, 2, -1, -1],       # A1
            [0, 0, 0, 4, -2, -2, 0, 0, 0, -2, 1, 1, 0, 0, 0, -2, 1, 1],         # E
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1, -1, 0, 0, 0, -2, 1, 1],          # E
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1],          # A2
            [0, 0, 0, 0, 2, -2, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1],          # E
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 1]            # E
        ]).T)

        # 33-35
        ch3_3rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],        # A2
            [2, 2, 2, -1, -1, -1, -1, -1, -1],  # E
            [0, 0, 0, 1, 1, 1, -1, -1, -1]      # E
        ]).T)

        Proj = block_diag(unc, cc_3str, ch3_3str, clc4_ang, ch3_3ang, ch3_3rot)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 4, 7, 13, 18, 21, 27],   # 8
            [10, 24, 30, 33],   # 4
            [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35],   # 24
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
