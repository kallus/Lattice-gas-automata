# -*- coding: utf-8 -*-
import random
import numpy as np
import hex
import c_module

N_STATES = 16
SQUARE_LATTICE = 0
HEX_LATTICE = 1
class LatticeModel(object):
    def __init__(self, lattice_size, n_particles, lattice_type):
        self.lattice_type = lattice_type
        self.size = lattice_size
        self.cells = np.zeros((self.size, self.size), dtype=np.int)
        self.cells_next = np.zeros((self.size, self.size), dtype=np.int)
        for i in xrange(n_particles):
            row = random.randint(0, lattice_size - 1)
            col = random.randint(0, lattice_size - 1)
            while self.cells[row, col] <> 0:
                row = random.randint(0, lattice_size - 1)
                col = random.randint(0, lattice_size - 1)
            self.cells[row, col] = random.choice([0b0100, 0b1000, 0b0001, 0b0010]) #0b0000, 
            #self.cells[row, col] = random.randint(0, N_STATES - 1)

    def update(self):
        if self.lattice_type == 0:
            c_module.c_module(self.cells, self.cells_next)
        elif self.lattice_type == 1:
            hex.hex(self.cells, self.cells_next)
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
