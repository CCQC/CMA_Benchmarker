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

        HA_str = np.array([
        [1, 1],
        [1,-1],
        ]).T

        P_combo = np.array([
        [1, 1],
        ]).T
        tor = np.array([
        [1, 1, 1, 1],
        ]).T

        M_combo = np.array([
        [1,-1],
        ]).T

        HOH_bend = np.array([
        [1],
        ]).T

        # Proj = block_diag(HA_str,HA_str,HOH_bend,HOH_bend,HOH_bend,HOH_bend,HA_str,P_combo,HOH_bend)
        Proj = block_diag(HA_str,HA_str,HOH_bend,HOH_bend,HOH_bend,HOH_bend,HOH_bend,HOH_bend,HOH_bend,P_combo)
        Proj = 1/norm(Proj,axis=0)*Proj
        
        self.Proj = Proj
        self.sym_sort = np.array([
            [0,1,2,4,5,6,7,8],
            [3,9,10,11],
            ],dtype=object)

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
