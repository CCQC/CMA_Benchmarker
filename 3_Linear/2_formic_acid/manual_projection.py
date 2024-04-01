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

        Unc = np.array([
        [1],
        ]).T

        Asym_Str = np.array([
        [1,  1],
        [1, -1],
        ]).T

        Asym_Bend = np.array([
        [1, -1,  1, -1],
        [1, -1, -1,  1],
        ]).T

        Proj = block_diag(Asym_Str,Asym_Str,Asym_Str,Asym_Str,Unc,Asym_Str,Asym_Str,Asym_Bend,Asym_Str,Asym_Str,Unc,Asym_Str,Asym_Str)
        Proj = 1/norm(Proj,axis=0)*Proj

        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)

