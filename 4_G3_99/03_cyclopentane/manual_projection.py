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

        a, b = np.cos(144*np.pi/180), np.cos(72*np.pi/180)
        c, d = np.sin(144*np.pi/180), np.sin(72*np.pi/180)

        cyc_5str = normalize(np.array([
            [1, 1, 1, 1, 1],
            [1, b, b, a, a],
            [0, d, -d, c, -c],
            [1, a, a, b, b],
            [0, c, -c, -d, d]
        ]).T)

        ch2_str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        cyc_5ang = normalize(np.array([
            [1,   a,   a,   b,   b],
            [0, a-b, b-a, 1-a, a-1]
        ]).T)

        ch2_ang = normalize(np.array([
            [4, -1, -1, -1, -1],
            [0, 1, 1, -1, -1],
            [0, 1, -1, 1, -1],
            [0, 1, -1, -1, 1]
        ]).T)

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

        cyc_5tor = normalize(np.array([
            [1,   b,   b,   a,   a],
            [0, 1-a, a-1, a-b, b-a]
        ]).T)

        Proj = block_diag(cyc_5str, ch2_str, ch2_2str, ch2_2str,
                          cyc_5ang, ch2_ang, ch2_2ang, ch2_2ang, cyc_5tor)

        self.Proj = Proj


def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
