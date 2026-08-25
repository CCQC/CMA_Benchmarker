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
        
        # 0, 41
        unc = np.eye(1)

        # 1-4
        cc_2str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        # 5-8
        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1,-1,-1],
            [1,-1, 1,-1],
            [1,-1,-1, 1]
        ]).T)

        # 9-14
        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 2,-1,-1],
            [2,-1,-1,-2, 1, 1],
            [0, 1,-1, 0, 1,-1],
            [0, 1,-1, 0,-1, 1]
        ]).T)

        # 15-16
        cc3_ang_2 = normalize(np.array([
            [2,-1,-1],
            [0, 1, -1]
        ]).T)

        # 17-18
        cc_2ang = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        # 19-26
        ch2_2ang = normalize(np.array([
            [4,-1,-1,-1,-1, 4,-1,-1,-1,-1],
            [4,-1,-1,-1,-1,-4, 1, 1, 1, 1],
            [0, 1, 1,-1,-1, 0, 1, 1,-1,-1],
            [0, 1, 1,-1,-1, 0,-1,-1, 1, 1],
            [0, 1,-1, 1,-1, 0, 1,-1, 1,-1],
            [0, 1,-1, 1,-1, 0,-1, 1,-1, 1],
            [0, 1,-1,-1, 1, 0, 1,-1,-1, 1],
            [0, 1,-1,-1, 1, 0,-1, 1, 1,-1]
        ]).T)

        # 27-36
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

        # 37-38
        cc_2rot = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1,-1,-1]
        ]).T)

        #39-40
        ch3_2rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1,-1,-1,-1]
        ]).T)

        Proj = block_diag(unc, cc_2str, cc_2str, ch2_2str, ch3_2str, cc3_ang_2, cc_2ang, ch2_2ang, ch3_2ang, cc_2rot, ch3_2rot, unc)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 3, 5, 9, 11, 15, 17, 19, 23, 27, 29, 33],
            [8, 14, 22, 26, 32, 36, 37, 39],
            [7, 13, 21, 25, 31, 35, 38, 40, 41],
            [2, 4, 6, 10, 12, 16, 18, 20, 24, 28, 30, 34],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
