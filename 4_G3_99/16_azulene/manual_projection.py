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
            [1,-1]
        ]).T)

        ch_2str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        a, b = np.cos(4*np.pi/5), np.cos(8*np.pi/5)
        c, d = np.sin(4*np.pi/5), np.sin(8*np.pi/5)

        cyc5_angtor = normalize(np.array([
            [1, a, b, b, a],
            [0, c, d,-d,-c]
        ]).T)

        a, b, c = -np.sin(np.pi/14), -np.cos(np.pi/7), np.sin(3*np.pi/14)
        d, e, f = np.cos(np.pi/14), -np.sin(np.pi/7), -np.cos(3*np.pi/14)

        cyc7_angtor = normalize(np.array([
            [1, a, b, c, c, b, a],
            [1, b, c, a, a, c, b],
            [0, d, e, f,-f,-e,-d],
            [0,-e, f, d,-d,-f, e]
        ]).T)

        ch_ang = normalize(np.array([
            [1,-1]
        ]).T)

        ch_2ang = normalize(np.array([
            [1,-1, 1,-1],
            [1,-1,-1, 1]
        ]).T)

        fus2_but = normalize(np.array([
            [1,-1]
        ]).T)

        ch_2oop = normalize(np.array([
            [1, 1],
            [1,-1],
        ]).T)

        Proj = block_diag(cc_2str, cc_2str, unc, cc_2str, cc_2str, cc_2str, unc, ch_2str, ch_2str, ch_2str, unc, cyc5_angtor, cyc7_angtor, ch_ang, ch_2ang, ch_2ang, ch_2ang, ch_ang, cyc5_angtor, cyc7_angtor, fus2_but, unc, ch_2oop, ch_2oop, ch_2oop, unc)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 18, 19, 21, 22, 26, 28, 30],
            [33, 35, 36, 41, 43, 45],
            [34, 37, 38, 39, 40, 42, 44, 46, 47],
            [1, 3, 6, 8, 10, 13, 15, 17, 20, 23, 24, 25, 27, 29, 31, 32],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
