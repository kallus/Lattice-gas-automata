# -*- coding: utf-8 -*-
import random
import numpy as np
import c_module
import pngnodes

N_STATES_SQUARE = 2**4
N_STATES_HEX = 2**6
SQUARE_LATTICE = 0
HEX_LATTICE = 1
class LatticeModel(object):
    def __init__(self, node_types, fraction_random_nodes, lattice_type):
        self.node_types = node_types
        self.lattice_type = lattice_type
        self.shape = node_types.shape
        self.cells = np.zeros(self.shape, dtype=np.int)
        self.cells_next = np.zeros(self.shape, dtype=np.int)

        #initialize particles only where there is not a wall
        n_non_wall_nodes = sum(sum(node_types != pngnodes.WALL))
        n_random_nodes = int(round(n_non_wall_nodes * fraction_random_nodes))
        
        for i in xrange(n_random_nodes):
            allowed = False
            while not allowed:
                row = random.randint(0, self.shape[0] - 1)
                col = random.randint(0, self.shape[1] - 1)
                allowed = ((pngnodes.WALL != node_types[row, col]) and 0 == self.cells[row, col])
            
            
            if SQUARE_LATTICE == self.lattice_type:
                max_state = N_STATES_SQUARE - 1
            elif HEX_LATTICE == self.lattice_type:
                max_state = N_STATES_HEX - 1
            self.cells[row, col] = random.randint(0, max_state)

    def update(self):
        if self.lattice_type == SQUARE_LATTICE:
            c_module.update4(self.cells, self.cells_next, self.node_types)
        elif self.lattice_type == HEX_LATTICE:
            c_module.update6(self.cells, self.cells_next, self.node_types)
        return
