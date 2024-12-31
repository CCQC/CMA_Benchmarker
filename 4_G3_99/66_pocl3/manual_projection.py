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

        c4_ang = normalize(np.array([
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        c4h_ang = normalize(np.array([
            [2, -1, -1],
            [0, 1, -1]
        ]).T)

        Proj = block_diag(unc, cc_3str, c4_ang, c4h_ang)

        self.Proj = Proj


def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
