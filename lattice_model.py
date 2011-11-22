# -*- coding: utf-8 -*-
import random
import numpy as np

N_STATES = 16
class LatticeModel(object):
    def __init__(self, lattice_size, n_particles):
        self.size = lattice_size
        self.cells = np.zeros((self.size, self.size), dtype=np.int)
        self.cells_next = np.zeros((self.size, self.size), dtype=np.int)
        for i in xrange(n_particles):
            row = random.randint(0, lattice_size - 1)
            col = random.randint(0, lattice_size - 1)
            while self.cells[row, col] <> 0:
                row = random.randint(0, lattice_size - 1)
                col = random.randint(0, lattice_size - 1)
            self.cells[row, col] = random.choice([0b0000, 0b0100, 0b1000, 0b0001, 0b0010])
            #self.cells[row, col] = random.randint(0, N_STATES - 1)

    def update(self):
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
