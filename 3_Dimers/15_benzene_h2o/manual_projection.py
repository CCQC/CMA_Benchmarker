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
        # 12-13
        Anti_sym_str = np.array([
        [1, 1],
        [1,-1],
        ]).T
        # 14,24-25,31
        Unc = np.array([
        [1],
        ]).T
        # 15-17
        HA_ang = normalize(np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T)
        # 18-23
        CH_ang = np.array([
        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [2,-2, 1,-1,-1, 1,-2, 2,-1, 1, 1,-1],
        [0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1],
        [2,-2,-1, 1,-1, 1, 2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1, 0, 0, 1,-1,-1, 1],
        ]).T
        # 26-27
        Anti_sym = np.array([
        [1,-1],
        ]).T
        # 28-30
        tor = normalize(np.array([
        [ 1,-1, 1,-1, 1,-1],
        [ 1, 0,-1, 1, 0,-1],
        [-1, 2,-1,-1, 2,-1],
        ]).T) 
        # 31,32
        Twist = np.array([
        [1, 1],
        ]).T
        # 33-38
        oop = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T

        # Proj = block_diag(HA_str,HA_str,Anti_sym_str,Unc,HA_ang,CH_ang,Unc,Unc,Anti_sym,tor,Twist,Unc,Twist,oop)
        Proj = block_diag(HA_str,HA_str,Anti_sym_str,Unc,HA_ang,CH_ang,Unc,Unc,Anti_sym,Anti_sym,tor,Twist,Twist,oop)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        # self.sym_sort = np.array([
            # [0,3,4,6,7,8,10,12,13,14,15,16,21,23,24,25,26,27,28,33,34,35,37], 
            # [1,2,5,9,11,17,18,19,20,22,29,30,31,32,36,38], 
            # ],dtype=object)
        
        # self.tiles = np.array([
            # # Benzene intramol. modes
            # [0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,28,29,30,33,34,35,36,37,38],
            # # Intermolecular modes
            # [14,25,26,27,31,32],
            # # H20 intramol. modes
            # [12,13,24],
            # ],dtype=object)

        # self.tile_type = ['m1','i','m2']
        # MP2/aTZ vals
        # self.tile_xi = {
                # 'm1m1'   :  0.1,
                # 'm2m2'   : 100.0,
                # 'ii'   :   0.05,
                # 'mi'   :  100.0,
                # 'm1m2' :  100.0,
                # }
        
        # MP2/haTZ vals
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

