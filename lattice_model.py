# -*- coding: utf-8 -*-
import random
import numpy as np
import c_module

N_STATES = 16
SQUARE_LATTICE = 0
HEX_LATTICE = 1
class LatticeModel(object):
    def __init__(self, lattice_size, n_particles, lattice_type):
        self.lattice_type = lattice_type
        self.size = lattice_size
        self.cells_next = np.zeros((self.size, self.size), dtype=np.int)
        self.node_type = np.zeros((self.size, self.size), dtype=np.int)

        # initialize cells randomly
        self.cells_temp = np.random.rand(self.size, self.size)
        if lattice_type == SQUARE_LATTICE:
          possible_directions = 4;
        else:
          possible_directions = 6;
        initial_density = 0.2;
        for i in xrange(possible_directions):
          self.cells_temp = np.where(self.cells_temp < (i+1)*1.0/possible_directions, 2**i, self.cells_temp)
        self.cells_temp = np.int_(self.cells_temp)
        mask = np.random.rand(self.size, self.size)
        mask = np.where(mask < initial_density, 1, 0)
        self.cells_temp = np.where(mask == 1, self.cells_temp, 0)
        self.cells_temp[0,:] = 0;
        self.cells_temp[:,0] = 0;
        self.cells_temp[self.size-1,:] = 0;
        self.cells_temp[:,self.size-1] = 0;
        self.cells = np.array(self.cells_temp, dtype=np.int)

    def update(self):
        if self.lattice_type == 0:
            c_module.update4(self.cells, self.cells_next, self.node_type)
        elif self.lattice_type == 1:
            c_module.update6(self.cells, self.cells_next, self.node_type)
        return
        # n_particles = 0
        # n_particles += (self.cells == 0b0001).sum()
        # n_particles += (self.cells == 0b0010).sum()
        # n_particles += (self.cells == 0b0100).sum()
        # n_particles += (self.cells == 0b1000).sum()
        # n_particles += (self.cells == 0b0011).sum()*2
        # n_particles += (self.cells == 0b0101).sum()*2
        # n_particles += (self.cells == 0b1001).sum()*2
        # n_particles += (self.cells == 0b0110).sum()*2
        # n_particles += (self.cells == 0b1010).sum()*2
        # n_particles += (self.cells == 0b1100).sum()*2
        # n_particles += (self.cells == 0b0111).sum()*3
        # n_particles += (self.cells == 0b1011).sum()*3
        # n_particles += (self.cells == 0b1101).sum()*3
        # n_particles += (self.cells == 0b1110).sum()*3
        # n_particles += (self.cells == 0b1111).sum()*4
        # print n_particles

        # propagation step.
        self.cells_next[:, :] = self.cells
        for row in xrange(self.size):
            if row == 0:
                row_minus_1 = self.size - 1
            else:
                row_minus_1 = row - 1
            if row == (self.size - 1):
                row_plus_1 = 0
            else:
                row_plus_1 = row + 1

            for col in xrange(self.size):
                if col == 0:
                    col_minus_1 = self.size - 1
                else:
                    col_minus_1 = col - 1
                if col == (self.size - 1):
                    col_plus_1 = 0
                else:
                    col_plus_1 = col + 1

                # ←
                if (self.cells[row, col] & 0b0100) <> 0:
                    self.cells_next[row, col] &= 0b1011
                    #assert((self.cells_next[row, col_minus_1] & 0b0001) == 0)
                    self.cells_next[row, col_minus_1] |= 0b0001
                # →
                if (self.cells[row, col] & 0b0001) <> 0:
                    self.cells_next[row, col] &= 0b1110
                    #assert((self.cells_next[row, col_plus_1] & 0b0100) == 0)
                    self.cells_next[row, col_plus_1] |= 0b0100
                # ↓
                if (self.cells[row, col] & 0b1000) <> 0:
                    self.cells_next[row, col] &= 0b0111
                    #assert((self.cells_next[row_plus_1, col] & 0b0010) == 0)
                    self.cells_next[row_plus_1, col] |= 0b0010
                # ↑
                if (self.cells[row, col] & 0b0010) <> 0:
                    self.cells_next[row, col] &= 0b1101
                    #assert((self.cells_next[row_minus_1, col] & 0b1000) == 0)
                    self.cells_next[row_minus_1, col] |= 0b1000

        self.cells[:, :] = 0
        for row in xrange(self.size):
            for col in xrange(self.size):
                # → ←
                if (self.cells_next[row, col] & 0b0101) == 0b0101:
                    # ↑
                    # ↓
                    self.cells[row, col] |= 0b1010
                elif (self.cells_next[row, col] & 0b0100) == 0b0100:
                    self.cells[row, col] |= 0b0001
                elif (self.cells_next[row, col] & 0b0001) == 0b0001:
                    self.cells[row, col] |= 0b0100

                if (self.cells_next[row, col] & 0b1010) == 0b1010:
                    self.cells[row, col] |= 0b0101
                elif (self.cells_next[row, col] & 0b1000) == 0b1000:
                    self.cells[row, col] |= 0b0010
                elif (self.cells_next[row, col] & 0b0010) == 0b0010:
                    self.cells[row, col] |= 0b1000
