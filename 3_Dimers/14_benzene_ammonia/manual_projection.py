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
        # 0-11
        HA_str = normalize(np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T)
        # 12-14
        Three_str = np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        # 15
        Unc = np.array([
        [1],
        ]).T
        # 16-18
        HA_ang = normalize(np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T)
        # 19-24
        CH_ang = np.array([
        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [2,-2, 1,-1,-1, 1,-2, 2,-1, 1, 1,-1],
        [0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1],
        [2,-2,-1, 1,-1, 1, 2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1, 0, 0, 1,-1,-1, 1],
        ]).T
        # 25-26,29-30
        NH_ang = np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        # 27-28 old
        # Anti_sym = np.array([
        # [1,-1],
        # ]).T
        # 27-28
        Anti_sym = np.array([
        [1,-1],
        ]).T

        # 31-33
        tor = normalize(np.array([
        [ 1,-1, 1,-1, 1,-1],
        [ 1, 0,-1, 1, 0,-1],
        [-1, 2,-1,-1, 2,-1],
        ]).T)
        
        # 34
        Anti_sym_6 = np.array([
        [1, 1, 1],
        ]).T
        
        # 35-40
        oop = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T
        # 41
        NH_oop = np.array([
        [1, 1, 1],
        ]).T


        Proj = block_diag(HA_str,HA_str,Three_str,Unc,HA_ang,CH_ang,NH_ang,Anti_sym,Anti_sym,NH_ang,tor,Anti_sym_6,oop,NH_oop)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        self.sym_sort = np.array([
            [0,3,4,6,7,8,10,12,13,15,16,17,22,24,25,27,29,31,32,35,36,37,39,41], 
            [1,2,5,9,11,14,18,19,20,21,23,26,28,30,33,34,38,40], 
        ],dtype=object)
        
        
        # self.tiles = np.array([
            # # Benzene intramol. modes
            # [0,1,2,3,4,5,6,7,9,10,11,16,17,18,19,20,21,22,23,24,32,33,34,35,36,37,38,39,40],
            # # Intermolecular modes
            # [15,27,28,29,30,31],
            # # Ammonia intramol. modes
            # [12,13,14,25,26,41],
            # ],dtype=object)

        # self.tile_type = ['m1','i','m2']
        
        
        # MP2/aTZ xi values
        # self.tile_xi = {
                # 'm1m1'   :  0.1,
                # 'm2m2'   :  100.0,
                # 'ii'   :  0.01,
                # 'mi'   :  100.0,
                # 'm1m2' :  100.0,
                # }
        # MP2/haTZ xi values
        # self.tile_xi = {
                # 'm1m1'   :  0.004,
                # 'm2m2'   :  100.0,
                # 'ii'   :   0.05,
                # 'mi'   :  100.0,
                # 'm1m2' :  100.0,
                # }

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

