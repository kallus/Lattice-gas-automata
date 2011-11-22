import numpy as np

N_STATES = 15
class LatticeModel(object):
    def __init__(self, lattice_size, n_particles):
        self.size = lattice_size
        self.cells = np.zeros((self.size, self.size), dtype=np.int)
        self.cells_next = np.zeros((self.size, self.size), dtype=np.int)
        for i in xrange(n_particles):
            row = random.randint(0, lattice_size)
            col = random.randint(0, lattice_size)
            while self.cells[row, col] <> 0:
                row = random.randint(0, lattice_size)
                col = random.randint(0, lattice_size)
            self.cells[row, col] = random.randint(0, N_STATES - 1)

    def update(self):
        # collision step.
        for row in xrange(self.lattice_size):
            for col in xrange(self.lattice_size):
                if self.cells[row, col] == 0:
                    #  .
                    # . .
                    #  .
                    # do nothing.
                    continue
                elif self.cells[row, col] == 1:
                    #  ↓
                    # . .
                    #  .
                    # no collision.
                    continue
                elif self.cells[row, col] == 2:
                    #  .
                    # . ←
                    #  .
                    # no collision.
                    continue
                elif self.cells[row, col] == 3:
                    #  ↓
                    # . ←
                    #  .
                    # no collision.
                    continue
                elif self.cells[row, col] == 4:
                    #  .
                    # . .
                    #  ↑
                    # no collision.
                    continue
                elif self.cells[row, col] == 5:
                    #  ↓
                    # . .
                    #  ↑
                    # => (10)
                    #  .
                    # → ←
                    #  .
                    # collision.
                    self.cells[row, col] == 10
                elif self.cells[row, col] == 6:
                    #  .
                    # . ←
                    #  ↑
                    # no collision.
                    continue
                elif self.cells[row, col] == 7:
                    #  ↓
                    # . ←
                    #  ↑
                    # collides, but maps to itself.
                    continue
                elif self.cells[row, col] == 8:
                    #  .
                    # → .
                    #  .
                    # no collision.
                    continue
                elif self.cells[row, col] == 9:
                    #  ↓
                    # → .
                    #  .
                    # collides, but maps to itself.
                    continue
                elif self.cells[row, col] == 10:
                    #  .
                    # → ←
                    #  .
                    # => (5)
                    #  ↓
                    # . .
                    #  ↑
                    # collision.
                    self.cells[row, col] = 5
                elif self.cells[row, col] == 11:
                    #  ↓
                    # → ←
                    #  .
                    # no collision.
                    continue
                elif self.cells[row, col] == 12:
                    #  .
                    # → .
                    #  ↑
                    # collides, but maps to itself.
                    continue
                elif self.cells[row, col] == 13:
                    #  ↓
                    # → .
                    #  ↑
                    # collides, but maps to itself.
                    continue
                elif self.cells[row, col] == 14:
                    #  .
                    # → ←
                    #  ↑
                    # collides, but maps to itself.
                    continue
                elif self.cells[row, col] == 15:
                    #  ↓
                    # → ←
                    #  ↑
                    # collides, but maps to itself.
                    continue

        # propagation step.
        for row in xrange(self.lattice_size):
            for col in xrange(self.lattice_size):

