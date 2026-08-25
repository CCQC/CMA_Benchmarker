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
        
        #0-5
        cc_6str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [2, 1, -1, -2, -1, 1],
            [0, 1, 1, 0, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)
        
        #6-9
        ch_4str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        #10-12
        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        #13-16
        ch_4ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1]
        ]).T)

        #17-19
        cyc_6tor = normalize(np.array([
            [1,-1, 1,-1, 1,-1],
            [2,-1,-1, 2,-1,-1],
            [0, 1,-1, 0, 1,-1]
        ]).T)

        #20-21
        cc_2tor = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        #22-22
        ch_4oop_sym = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1]
        ]).T)

        Proj = block_diag(cc_6str, ch_4str, cyc_6ang,
                          ch_4ang, cyc_6tor, cc_2tor, ch_4oop_sym)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 4, 6, 11, 13],
            [5, 9, 12, 16],
            [17, 21],
            [23],
            [18, 20],
            [19, 22],
            [1, 2, 8, 15],
            [3, 7, 10, 14],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
