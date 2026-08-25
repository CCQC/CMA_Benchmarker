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

        cc_6str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [2, 1, -1, -2, -1, 1],
            [0, 1, 1, 0, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
        ]).T)

        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        ch_2str = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        cyc_6ang = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [2, -1, -1, 2, -1, -1],
            [0, 1, -1, 0, 1, -1],
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

        ch_2ang = normalize(np.array([
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        cyc_6tor = normalize(np.array([
            [1, -1, 1, -1, 1, -1],
            [0, -1, 1, 0, -1, 1],
            [2, -1, -1, 2, -1, -1],
        ]).T)

        ch_2oop = normalize(np.array([
            [1, 1],
            [1, -1],
        ]).T)

        Proj = block_diag(cc_6str, ch2_2str, ch_2str, ch_2str, cyc_6ang,
                          ch2_2ang, ch_2ang, ch_2ang, cyc_6tor, ch_2oop, ch_2oop)

        self.Proj = Proj

        self.sym_sort = np.array([
            [0, 1, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27, 29, 31, 32, 34],
            [3, 5, 7, 9, 11, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35],
        ], dtype=object)

def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
