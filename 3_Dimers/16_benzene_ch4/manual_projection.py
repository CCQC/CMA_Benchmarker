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
        # 12
        Unc = np.array([
        [1],
        ]).T
        # 13-15,16-18
        Three_str = np.array([
        [1, 1, 1],
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        # 19-21
        HA_ang = normalize(np.array([
        [1,-1, 1,-1, 1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T)
        # 22-27
        CH_ang = np.array([
        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [2,-2, 1,-1,-1, 1,-2, 2,-1, 1, 1,-1],
        [0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1],
        [2,-2,-1, 1,-1, 1, 2,-2,-1, 1,-1, 1],
        [0, 0, 1,-1,-1, 1, 0, 0, 1,-1,-1, 1],
        ]).T
        # 28-32
        Six_ang = np.array([
        [1, 1, 1,-1,-1,-1],
        [2,-1,-1, 0, 0, 0],
        [0, 1,-1, 0, 0, 0],
        [0, 0, 0, 2,-1,-1],
        [0, 0, 0, 0, 1,-1],
        ]).T
        # 33-34
        Three_ang = np.array([
        [2,-1,-1],
        [0, 1,-1],
        ]).T
        # 
        Anti_sym = np.array([
        [1,-1],
        ]).T
        # 35-37
        tor = normalize(np.array([
        [ 1,-1, 1,-1, 1,-1],
        [ 1, 0,-1, 1, 0,-1],
        [-1, 2,-1,-1, 2,-1],
        ]).T)
        # 38
        Twist = np.array([
        [1, 1, 1],
        ]).T
        # 39-44
        oop = np.array([
        [1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1],
        [2, 1,-1,-2,-1, 1],
        [0, 1, 1, 0,-1,-1],
        [2,-1,-1, 2,-1,-1],
        [0, 1,-1, 0, 1,-1],
        ]).T


        # Proj = block_diag(HA_str,HA_str,str1,Unc,HA_ang,CH_ang,ang,Unc,Anti_sym,tor,Twist,Unc,Twist,oop)
        Proj = block_diag(HA_str,HA_str,Unc,Three_str,Three_str,HA_ang,CH_ang,Six_ang,Three_ang,tor,Twist,oop)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj
        # self.sym_sort = np.array([
           # [0,3,4,6,7,8,10,12,13,14,16,17,18,23,25,26,27,29,31,33,35,36,39,40,41,43], # a'
           # [1,2,5,9,11,15,19,20,21,22,24,28,30,32,34,37,38,42,44], # a"
           # # [41,42,43,44], # e
           # ],dtype=object)
        # Cs symsort
        self.sym_sort = np.array([
           [0,3,4,6,7,8,10,12,13,14,16,17,19,20,25,27,28,29,31,33,35,36,39,40,41,43], # a'
           [1,2,5,9,11,15,18,21,22,23,24,26,30,32,34,37,38,42,44], # a"
           ],dtype=object)
        
        # self.tiles = np.array([
            # # Benzene intramol. modes
            # [0,1,2,3,4,5,6,7,8,9,10,11,17,18,19,20,21,22,23,24,25,35,36,37,39,40,41,42,43,44],
            # # Intermolecular modes
            # [16,31,32,33,34,38],
            # # CH4 intramol. modes
            # [12,13,14,15,26,27,28,29,30],
            # ],dtype=object)

        # self.tile_type = ['m1','i','m2']
        
        # MP2/aTZ xi values
        # self.tile_xi = {
                # 'm1m1'   :  0.1,
                # 'm2m2'   :  100.0,
                # 'ii'   :  1.0,
                # 'mi'   :  100.0,
                # 'm1m2' :  100.0,
                # }
        # MP2/haTZ xi values
        # self.tile_xi = {
                # 'm1m1'   :  0.004,
                # 'm2m2'   :  100.0,
                # 'ii'   :  100.0,
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
   
