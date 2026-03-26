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

        a, b = np.cos(144*np.pi/180), np.cos(72*np.pi/180)
        c, d = np.sin(144*np.pi/180), np.sin(72*np.pi/180)
        # 0-4
        HA_str = normalize(np.array([
        [1, 1, 1, 1, 1],
        [1, b, b, a, a],
        [0, d,-d, c,-c],
        [1, a, a, b, b],
        [0, c,-c,-d, d],
        ]).T)
        # HA_str1 = np.eye(1)
       
        # HA_str2 = normalize(np.array([
        # [1, 1],
        # [1,-1],
        # ]).T)
       
        # HA_str3 = normalize(np.array([
        # [1, 1],
        # [1,-1],
        # ]).T)
        # 5-6
        CH_str1 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
        # 7-8
        CH_str2 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
        # 9-10
        HA_ang = normalize(np.array([
        [1,   a,   a,   b,   b],
        [0, a-b, b-a, 1-a, a-1],
        ]).T)
        # 11-12
        CH_ang1 = normalize(np.array([
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)
        # 13-14
        CH_ang2 = normalize(np.array([
        [1,-1, 1,-1],
        [1,-1,-1, 1],
        ]).T)
        # 15-16
        tor = normalize(np.array([
        [1,   b,   b,   a,   a],
        [0, 1-a, a-1, a-b, b-a],
        ]).T)
        # 17-18
        oop2 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)
        # 19-20
        oop3 = normalize(np.array([
        [1, 1],
        [1,-1],
        ]).T)

        Proj = block_diag(HA_str,CH_str1,CH_str2,HA_ang,CH_ang1,CH_ang2,tor,oop2,oop3)
        # Proj = block_diag(HA_str1,HA_str2,HA_str3,CH_str1,CH_str2,HA_ang,CH_ang1,CH_ang2,tor,oop2,oop3)

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,3,5,7,9,11,13],
            [15,18,20],
            [16,17,19],
            [2,4,6,8,10,12,14],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

