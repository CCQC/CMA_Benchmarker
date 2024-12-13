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

        HA_stretch = np.eye(2)
        
        stretch = normalize(np.array([
        [ 1, 1],
        [ 1,-1],
        ]).T)

        angles = normalize(np.array([
        [ 1, 0, 0, 0],
        [ 0, 2,-1,-1],
        [ 0, 0, 1,-1],
        ]).T)

        tor = normalize(np.array([
        [ 1, 1],
        ]).T)

        Proj = block_diag(stretch,HA_stretch,angles,tor,np.eye(1))
        # self.sym_sort = np.array([
            # [0,1,2,3,4,5,6],
            # [7,8],
            # ],dtype=object)


        self.Proj = Proj

def normalize(mat):
    return 1/norm(mat,axis=0)*mat

if __name__=="__main__":
    np.set_printoptions(linewidth=400, precision=2,threshold=100000)
    p = Projection([])
    p.run()
    print(p.Proj)
