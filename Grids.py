import numpy as np

class MPMGrid1D:
    def __init__(self, lmin, lmax, NoCells):
        self.lmin = lmin
        self.lmax = lmax
        self.dx   = (lmax-lmin)/NoCells
        self.NoCells = NoCells
        self.NoNodes = NoCells +1

    def GridNodes(self, x_P):
        # compute left and right nodal indexes
        left_node_id  = np.floor(x_P/self.dx)
        right_node_id = left_node_id + 1
        return np.array([left_node_id, right_node_id], dtype=int)

    def GridNodeX(self, NodeIDs):
        # compute left and right nodal coordinates
        return np.array([i*self.dx for i in NodeIDs])