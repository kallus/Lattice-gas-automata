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
        if lattice_type == SQUARE_LATTICE:
            print "init"
            c_module.init_cells(self.cells, node_types, 15)
            self.cells_temp = np.array(self.cells)
            possible_directions = 4
        else:
            c_module.init_cells(self.cells, node_types, 63)
            self.cells_temp = np.array(self.cells)
            possible_directions = 6

#             self.cells_temp = np.random.rand(self.shape[0], self.shape[1])
#             random_particles = np.multiply(node_types > 0, node_types)
#             has_particle = (np.random.randint(0, 255, self.shape) < random_particles)
#             for i in xrange(possible_directions):
#                 self.cells_temp = np.where(self.cells_temp < (i+1)*1.0/possible_directions, 2**i, self.cells_temp)
#             self.cells_temp = np.int_(self.cells_temp)
# #        mask = has_particle
#             mask = np.random.rand(self.shape[0], self.shape[1]) * 255
#             mask = np.where(mask < node_types, 1, 0)
#             self.cells_temp = np.where(mask == 1, self.cells_temp, 0)
#             self.cells_temp = np.where(node_types == pngnodes.WALL, 0, self.cells_temp)
#             self.cells = np.array(self.cells_temp, dtype=np.int)

    def update(self):
        if self.lattice_type == SQUARE_LATTICE:
            c_module.update4(self.cells, self.cells_next, self.node_types, self.cell_colors,
                             self.temperature)
        elif self.lattice_type == HEX_LATTICE:
            c_module.update6(self.cells, self.cells_next, self.node_types, self.cell_colors,
                             self.temperature)
        return

    def add_particles(self, pos):
        radius = 20
        xmin = max(0, pos[0]-radius)
        ymin = max(0, pos[1]-radius)
        xmax = min(self.shape[0], pos[0]+radius)
        ymax = min(self.shape[1], pos[1]+radius)

        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                if (i-pos[0])**2 + (j-pos[1])**2 < radius**2:
                    if HEX_LATTICE == self.lattice_type:
                        self.cells[i, j] = N_STATES_HEX - 1
                    elif SQUARE_LATTICE == self.lattice_type:
                        self.cells[i, j] = N_STATES_SQUARE - 1

    def remove_particles(self, pos):
        radius = 40
        xmin = max(0, pos[0]-radius)
        ymin = max(0, pos[1]-radius)
        xmax = min(self.shape[0], pos[0]+radius)
        ymax = min(self.shape[1], pos[1]+radius)

        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                if (i-pos[0])**2 + (j-pos[1])**2 < radius**2:
                    self.cells[i, j] = 0
