import os
import sys
import time
import numpy as np
import pygame
import pygame.display
import pygame.event
import scipy.ndimage
from lattice_model import LatticeModel
import pngnodes

class PygameView(object):
    def __init__(self, lattice_model, delay):
        print "init"
        self.fps = 30
        self.lattice_model = lattice_model
        self.pixmap = np.ones((lattice_model.shape[0], lattice_model.shape[1], 3), dtype=np.uint8)
        self.transpose = np.zeros((lattice_model.shape[1], lattice_model.shape[0], 3), dtype=np.uint8)
        self.transpose2d = np.zeros((lattice_model.shape[1], lattice_model.shape[0]), dtype=np.uint32)
#        pygame.event.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode([lattice_model.shape[1], lattice_model.shape[0]]) #, pygame.HWSURFACE | pygame.FULLSCREEN)
#        tick_time = clock.tick(fps)
        pygame.display.set_caption("Lattice gas")
        self.wallmap = np.ones((lattice_model.shape[1], lattice_model.shape[0], 3), dtype=np.uint8)
        self.wallmap.fill(0)
        for i in xrange(lattice_model.shape[0]):
          for j in xrange(lattice_model.shape[1]):
            if lattice_model.node_types[i,j]==pngnodes.WALL:
              self.wallmap[j,i,0] = 12
              self.wallmap[j,i,1] = 82
              self.wallmap[j,i,2] = 232
        self.wallsurface = pygame.surfarray.make_surface(self.wallmap)
        self.wallsurface.set_colorkey((0, 0, 0))
        while True:
            if (pygame.event.peek(pygame.KEYDOWN)):
                exit(1)
            self.update()

    def update(self):
        #print "update"
#        self.pixmap[0:20, :, :] = 255
#        start = time.time()

        nsteps = 5
        for i in xrange(nsteps):
            self.lattice_model.update()
#        stop = time.time()
#        print("executed %i steps in %f milliseconds" % (nsteps, stop - start,))
#        exit(1)
        #print self.lattice_model.cells

        if self.lattice_model.lattice_type == 0:
            self.pixmap.fill(255)
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0001)*128
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0010)*128
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0100)*128
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1000)*128

            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1001)*159
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0101)*159
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0011)*159
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1010)*159
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0110)*159
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1100)*159

            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b0111)*223
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1011)*223
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1101)*223
            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1110)*223

            self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b1111)*255
#         elif self.lattice_model.lattice_type == 1:
# #            self.pixmap[:, :, 0] -= (self.lattice_model.cells != 0b000000)*255

#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100000)*128
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010000)*128
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001000)*128
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000100)*128
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000010)*128
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000001)*128

#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110000)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101000)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100100)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100010)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100001)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011000)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010100)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010010)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010001)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001100)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001010)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001001)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000110)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000101)*154
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000011)*154

#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111000)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110100)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110010)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110001)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101100)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101010)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101001)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100110)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100101)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100011)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011100)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011010)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011001)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010110)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010101)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010011)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001110)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001101)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001011)*179
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000111)*179

#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111100)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111010)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111001)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110110)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110101)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110011)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101110)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101101)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101011)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100111)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011110)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011101)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011011)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010111)*205
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001111)*205

#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111110)*230
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111101)*230
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111011)*230
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110111)*230
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101111)*230
#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011111)*230

#             self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111111)*255

#        scipy.ndimage.filters.gaussian_filter(self.lattice_model.cell_colors, 0.75,
#                                              output=self.lattice_model.cell_colors)
        self.transpose2d = self.lattice_model.cell_colors.T
#        self.transpose[:, :, 1] = arr
#        self.transpose[:, :, 2] = arr
        pygame.surfarray.blit_array(self.screen, self.transpose2d)
#        self.screen.get_buffer().write(self.lattice_model.cell_colors.tostring(), 0)
#        self.screen.fill((255, 255, 255))

        # show walls
        self.screen.set_colorkey((0, 0, 0))
        self.screen.blit(self.wallsurface, (0, 0))

#        pygame.display.update()
        pygame.display.flip()

if __name__ == "__main__":
    print "main"

    if len(sys.argv) != 2:
        print("Please specify map file.")
        exit(1)
    elif os.path.isfile(sys.argv[1]) == False:
        print("Map file name %s is not valid" % (sys.argv[1],))
    
    node_types = pngnodes.read(sys.argv[1])
    height = node_types.shape[0]
    width = node_types.shape[1]

    density = 0.5

    model = LatticeModel(node_types, density, 1, 0)

#    model.cells[0:height*5/6, 0:width*5/6] = 0
#    model.cells[:, width*5/9:] = 0
    view = PygameView(model, 10)

