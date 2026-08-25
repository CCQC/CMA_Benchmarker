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

        # 0
        unc = np.eye(1)

        # 1-6
        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1], # A1g
            [1, 1, 1,-1,-1,-1], # A2u
            [2,-1,-1, 2,-1,-1], # Eg
            [2,-1,-1,-2, 1, 1], # Eu
            [0, 1,-1, 0, 1,-1], # Eg
            [0, 1,-1, 0,-1, 1]  # Eu
        ]).T)

        # 7-16
        ch3_2ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1],   # A1g
            [1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1],   # A2u
            [2,-1,-1, 0, 0, 0, 2,-1,-1, 0, 0, 0],   # Eg
            [2,-1,-1, 0, 0, 0,-2, 1, 1, 0, 0, 0],   # Eu
            [0, 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0],   # Eg
            [0, 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0],   # Eu
            [0, 0, 0, 2,-1,-1, 0, 0, 0, 2,-1,-1],   # Eg
            [0, 0, 0, 2,-1,-1, 0, 0, 0,-2, 1, 1],   # Eu
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1],   # Eg
            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1]    # Eu
        ]).T)

        # 17
        ch3_rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1] # A1u
        ]).T)

        Proj = block_diag(unc, ch3_2str, ch3_2ang, ch3_rot)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 7],
            [],
            [3, 5, 9, 11, 13, 15],
            [17],
            [2, 8],
            [4, 6, 10, 12, 14, 16],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
