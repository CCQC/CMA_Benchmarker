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

        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [2, -1, -1, -2, 1, 1],
            [0, 1, -1, 0, 1, -1],
            [0, 1, -1, 0, -1, 1]
        ]).T)

        # c5_ang = normalize(np.array([
        #     [2, 2,-1,-1,-1,-1],
        #     [0, 0, 1,-1,-1, 1],
        #     [1,-1, 0, 0, 0, 0],
        #     [0, 0, 1, 0, 0,-1],
        #     [0, 0, 0, 1,-1, 0],
        # ]).T)

        c2so2_ang = normalize(np.array([
            [1,-1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1,-1,-1, 1],
            [0, 0, 1, 1,-1,-1],
            [0, 0, 1,-1, 1,-1],
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

        ch3_2rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1]
        ]).T)

        Proj = block_diag(cc_2str, cc_2str, ch3_2str,
                          c2so2_ang , ch3_2ang, ch3_2rot)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 2, 4, 6, 10, 11, 15, 17, 21],
            [9, 12, 20, 24, 25],
            [1, 8, 13, 19, 23, 26],
            [3, 5, 7, 14, 16, 18, 22],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
