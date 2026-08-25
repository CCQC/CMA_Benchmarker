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
        
        # 5, 35
        unc = np.eye(1)

        a, b = np.cos(144*np.pi/180), np.cos(72*np.pi/180)
        c, d = np.sin(144*np.pi/180), np.sin(72*np.pi/180)

        # 0-4
        cyc_5str = normalize(np.array([
            [1, 1, 1, 1, 1],
            [1, b, b, a, a],
            [0, d, -d, c, -c],
            [1, a, a, b, b],
            [0, c, -c, -d, d],
        ]).T)

        # 6-9, 10-13
        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        # 14-15
        cyc_5ang = normalize(np.array([
            [1,   a,   a,   b,   b],
            [0, a-b, b-a, 1-a, a-1],
        ]).T)

        # 16
        ch_ang = normalize(np.array([
            [1, -1]
        ]).T)

        # 17-24, 25-32
        ch2_2ang = normalize(np.array([
            [4, -1, -1, -1, -1, 4, -1, -1, -1, -1],
            [4, -1, -1, -1, -1, -4, 1, 1, 1, 1],
            [0, 1, 1, -1, -1, 0, 1, 1, -1, -1],
            [0, 1, 1, -1, -1, 0, -1, -1, 1, 1],
            [0, 1, -1, 1, -1, 0, 1, -1, 1, -1],
            [0, 1, -1, 1, -1, 0, -1, 1, -1, 1],
            [0, 1, -1, -1, 1, 0, 1, -1, -1, 1],
            [0, 1, -1, -1, 1, 0, -1, 1, 1, -1]
        ]).T)

        # 33-34
        cyc_5tor = normalize(np.array([
            [1,   b,   b,   a,   a],
            [0, 1-a, a-1, a-b, b-a],
        ]).T)

        Proj = block_diag(cyc_5str, unc, ch2_2str, ch2_2str,
                          cyc_5ang, ch_ang, ch2_2ang, ch2_2ang, cyc_5tor, unc)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 3, 5, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31, 33],
            [2, 4, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
