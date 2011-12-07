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
    def __init__(self, node_types, fraction_random_nodes, lattice_type, temperature):
        self.node_types = node_types
        self.lattice_type = lattice_type
        self.shape = node_types.shape
        self.temperature = temperature;
        self.cells = np.zeros(self.shape, dtype=np.int)
        self.cells_next = np.zeros(self.shape, dtype=np.int)
        self.cell_colors = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint32)
        self.cell_colors.fill((232 << 24) | (82 << 16) | (12 << 8))

        # initialize cells randomly
        self.cells_temp = np.random.rand(self.shape[0], self.shape[1])
        if lattice_type == SQUARE_LATTICE:
          possible_directions = 4;
        else:
          possible_directions = 6;
        
        random_particles = np.multiply(node_types > 0, node_types)
        has_particle = (np.random.randint(0, 255, self.shape) < random_particles)
        for i in xrange(possible_directions):
          self.cells_temp = np.where(self.cells_temp < (i+1)*1.0/possible_directions, 2**i, self.cells_temp)
        self.cells_temp = np.int_(self.cells_temp)
#        mask = has_particle
        mask = np.random.rand(self.shape[0], self.shape[1]) * 255
        mask = np.where(mask < node_types, 1, 0)
        self.cells_temp = np.where(mask == 1, self.cells_temp, 0)
        self.cells_temp = np.where(node_types == pngnodes.WALL, 0, self.cells_temp)
        self.cells = np.array(self.cells_temp, dtype=np.int)

    def update(self):
        if self.lattice_type == SQUARE_LATTICE:
            c_module.update4(self.cells, self.cells_next, self.node_types, self.cell_colors)
        elif self.lattice_type == HEX_LATTICE:
            c_module.update6(self.cells, self.cells_next, self.node_types, self.cell_colors,
                             self.temperature)
        return
