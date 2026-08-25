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

        # 0-1
        so_2str = normalize(np.array([
            [1, 1], # A1
            [1,-1]  # B2
        ]).T)

        # 2-3
        scl_2str = normalize(np.array([
            [1, 1], # A1
            [1,-1]  # B1
        ]).T)

        # 4, 5
        unc = np.eye(1)

        # 6-8
        cl2so2_ang = normalize(np.array([
            [1,-1,-1, 1], # A2
            [1, 1,-1,-1], # B1
            [1,-1, 1,-1], # B2
        ]).T)

        Proj = block_diag(so_2str, scl_2str, unc, unc, cl2so2_ang)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 2, 4, 5],
            [6],
            [1, 7],
            [3, 8],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
