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
        
        # 0-5, 9, 18, 39, 40, 44
        unc = np.eye(1)

        # 6-8, 10-12, 13-15
        ch3_str = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 16-17, 
        c4_planeang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 19-21
        c4_ang = normalize(np.array([
            [1, 1, 1],
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 22-26, 29-33, 34-38
        ch3_ang = normalize(np.array([
            [1, 1, 1,-1,-1,-1],
            [2,-1,-1, 0, 0, 0],
            [0, 1,-1, 0, 0, 0],
            [0, 0, 0, 2,-1,-1],
            [0, 0, 0, 0, 1,-1]
        ]).T)

        # 27-28
        hc4_ang = normalize(np.array([
            [2,-1,-1],
            [0, 1,-1]
        ]).T)

        # 41
        cco_rot = normalize(np.array([
            [1, 1]
        ]).T)

        # 42, 43
        ch3_rot = normalize(np.array([
            [1, 1, 1, 1, 1, 1]
        ]).T)

        Proj = block_diag(unc, unc, unc, unc, unc, unc, ch3_str, unc, ch3_str, ch3_str, c4_planeang, unc, c4_ang, ch3_ang, hc4_ang, ch3_ang, ch3_ang, unc, cco_rot, ch3_rot, ch3_rot, ch3_rot, unc)

        self.Proj = Proj


def normalize(mat):
    return 1/norm(mat, axis=0)*mat


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=2, threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)