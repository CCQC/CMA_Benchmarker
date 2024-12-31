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
            [1, -1]
        ]).T)

        ch2_2str = normalize(np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ]).T)

        ch3_2str = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 2, -1, -1],
            [2, -1, -1, -2, 1, 1],
            [0, 1, -1, 0, 1, -1],
            [0, 1, -1, 0, -1, 1]
        ]).T)

        cc_2ang = normalize(np.array([
            [1, 1],
            [1, -1]
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

        ch3_2ang = normalize(np.array([
            [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
            [1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1],
            [2, -1, -1, 0, 0, 0, 2, -1, -1, 0, 0, 0],
            [2, -1, -1, 0, 0, 0, -2, 1, 1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 2, -1, -1, 0, 0, 0, 2, -1, -1],
            [0, 0, 0, 2, -1, -1, 0, 0, 0, -2, 1, 1],
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1],
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 1]
        ]).T)

        cc_2tor = normalize(np.array([
            [1, 1],
            [1, -1]
        ]).T)

        ch3_2rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1]
        ]).T)

        Proj = block_diag(unc, cc_2str, cc_2str, cc_2str, ch2_2str, ch2_2str, ch2_2str, ch3_2str, cc_2ang,
                          cc_2ang, cc_2ang, ch2_2ang, ch2_2ang, ch2_2ang, ch3_2ang, cc_2tor, cc_2tor, unc, ch3_2rot)

        self.Proj = Proj
        '''
        self.sym_sort = np.array([                   
    		[],
    		[]              
    		],dtype=object)                          
	'''


def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
