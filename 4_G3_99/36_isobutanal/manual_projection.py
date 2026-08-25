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
        
        # 3, 7, 11, 15, 32
        unc = np.eye(1)

        # 0-2
        cc_3str = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1],
        ]).T)

        # 4-6, 8-10
        ch_3str = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 12-14
        c4_ang = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 16-20, 21-25
        ch3_ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1],
            [0, 0, 0, 0, 1,-1]
        ]).T)

        # 26-27
        c4h_ang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 28
        ch_ang = normalize(np.array([
            [1,-1]
        ]).T)

        # 29
        ccc3_rot = normalize(np.array([
            [1, 1]
        ]).T)

        # 30, 31
        ch3_rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1]
        ]).T)

        Proj = block_diag(cc_3str, unc, ch_3str, unc, ch_3str, unc, c4_ang, unc, ch3_ang, ch3_ang, c4h_ang, ch_ang, ccc3_rot, ch3_rot, ch3_rot, unc)

        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
