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
            [2,-1,-1],
            [0, 1,-1],
        ]).T)

        ch3_str = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 2,-1,-1],
            [2,-1,-1,-2, 1, 1],
            [0, 1,-1, 0, 1,-1],
            [0, 1,-1, 0,-1, 1]
        ]).T)

        c4_ang = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        oc4_ang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        ch3_ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1],
            [0, 0, 0, 0, 1,-1]
        ]).T)

        ch3_2ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1],
            [1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
            [2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0],
            [2,-1,-1, 0, 0, 0,-2, 1, 1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1],
            [0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1],
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1],
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1]
        ]).T)

        ch3_rot2 = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).T)

        ch3_rot = normalize(np.array([
            [1, 1, 1]
        ]).T)

        ch3_2rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        ]).T)

        Proj = block_diag(cc_3str, unc, unc, ch3_str, ch3_str, ch3_2str, c4_ang, unc, oc4_ang, ch3_ang, ch3_ang, ch3_2ang, ch3_rot, ch3_rot2, ch3_rot, ch3_2rot)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17, 18, 20, 21, 23, 24, 26, 28, 29, 31, 33, 35, 37, 39, 41, 47],
            [2, 7, 10, 12, 14, 16, 19, 22, 25, 27, 30, 32, 34, 36, 38, 40, 42, 43, 44, 45, 46],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
