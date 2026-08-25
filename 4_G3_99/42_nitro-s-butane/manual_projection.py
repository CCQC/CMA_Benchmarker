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

        no_2str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        cc_3str = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1],
        ]).T)

        ch2_str = normalize(np.array([
            [1, 1],
            [1,-1]
        ]).T)

        ch3_str = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        ch2c_ang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        c4_ang = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        hc4_ang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        ch2_ang = normalize(np.array([
            [4,-1,-1,-1,-1],
            [0, 1, 1,-1,-1],
            [0, 1,-1, 1,-1],
            [0, 1,-1,-1, 1]
        ]).T)

        ch3_ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1],
            [0, 0, 0, 0, 1,-1]
        ]).T)

        ch2c_rot = normalize(np.array([
            [1, 1, 1, 1]
        ]).T)

        ch3_rot1 = normalize(np.array([
            [1, 1, 1]
        ]).T)

        ch3_rot2 = normalize(np.array([
            [1, 1, 1, 1, 1, 1]
        ]).T)

        Proj = block_diag(no_2str, cc_3str, unc, unc, ch2_str, ch3_str, ch3_str, ch2c_ang, c4_ang, unc, hc4_ang, ch2_ang, ch3_ang, ch3_ang, ch2c_rot, unc, ch3_rot1, ch3_rot2, unc)

        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
