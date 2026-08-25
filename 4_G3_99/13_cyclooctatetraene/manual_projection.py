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

        a = 1/(2**0.5)

        # 0-7
        cyc8_str = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],   # A1 Isolated to str
            [1,-1, 1,-1, 1,-1, 1,-1],   # A1
            [0, 1, 0,-1, 0, 1, 0,-1],   # B1
            [1, 0,-1, 0, 1, 0,-1, 0],   # B2
            [1, a, 0,-a,-1,-a, 0, a],   # E Both E isolated to str
            [0, a, 1, a, 0,-a,-1,-a],
            [1,-a, 0, a,-1, a, 0,-a],   # E
            [0, a,-1, a, 0,-a, 1,-a]
        ]).T)

        # 8-15
        ch_8str = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],   # a1 Isolated to str
            [1,-1, 1,-1, 1,-1, 1,-1],   # a2
            [1,-1,-1, 1, 1,-1,-1, 1],   # b1  Had to do m combo for sym
            [1, 1,-1,-1, 1, 1,-1,-1],   # b2  Had to do p combo for sym 
            [1, a, 0,-a,-1,-a, 0, a],   # e Both E isolated to str
            [0, a, 1, a, 0,-a,-1,-a],
            [1,-a, 0, a,-1, a, 0,-a],   # e
            [0, a,-1, a, 0,-a, 1,-a]
        ]).T)

        # 16-20
        cyc8_ang = normalize(np.array([
            [1,-1, 1,-1, 1,-1, 1,-1],   # a2
            [1, 1,-1,-1, 1, 1,-1,-1],   # b2  Had to do p combo for sym 
            [1,-1,-1, 1, 1,-1,-1, 1],   # b1  Had to do m combo for sym
            [1,-a, 0, a,-1, a, 0,-a],   # e
            [0, a,-1, a, 0,-a, 1,-a]
        ]).T)

        # 21-28
        ch_8ang = normalize(np.array([
            [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],   # A1
            [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],   # A2
            [1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1],   # B2
            [1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1],   # B1
            [1,-1,-a, a, 0, 0, a,-a,-1, 1, a,-a, 0, 0,-a, a],   # E
            [0, 0, a,-a,-1, 1, a,-a, 0, 0,-a, a, 1,-1,-a, a],
            [a,-a, 1,-1, a,-a, 0, 0,-a, a,-1, 1,-a, a, 0, 0],   # E
            [-a, a, 0, 0, a,-a, 1,-1, a,-a, 0, 0,-a, a,-1, 1]
        ]).T)

        # 29-33
        cyc8_tor = normalize(np.array([
            [1,-1, 1,-1, 1,-1, 1,-1],   # B1
            [1, 0,-1, 0, 1, 0,-1, 0],   # A2
            [0, 1, 0,-1, 0, 1, 0,-1],   # A1
            [1,-a, 0, a,-1, a, 0,-a],   # E
            [0, a,-1, a, 0,-a, 1,-a]
        ]).T)

        # 34-41
        ch_8oop = normalize(np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],   # B2
            [1,-1, 1,-1, 1,-1, 1,-1],   # B1
            [1,-1,-1, 1, 1,-1,-1, 1],   # A2
            [1, 1,-1,-1, 1, 1,-1,-1],   # A1
            [1, a, 0,-a,-1,-a, 0, a],   # E
            [0, a, 1, a, 0,-a,-1,-a],
            [1,-a, 0, a,-1, a, 0,-a],   # E
            [0, a,-1, a, 0,-a, 1,-a]
        ]).T)

        Proj = block_diag(cyc8_str, ch_8str, cyc8_ang, ch_8ang, cyc8_tor, ch_8oop)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0,1,8,21,31,37], # a1
            [9,16,22,30,36], # a2
            [2,10,18,24,29,35], # b1
            [3,11,17,23,34], # b2
            [4,5,6,7,12,13,14,15,19,20,25,26,27,28,32,33,38,39,40,41], # e
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
