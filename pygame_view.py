import sys
import numpy as np
import pygame
import pygame.display
import scipy.ndimage
from lattice_model import LatticeModel
import pngnodes

class PygameView(object):
    def __init__(self, lattice_model, delay):
        print "init"
        self.fps = 30
        self.lattice_model = lattice_model
        self.pixmap = np.ones((lattice_model.shape[0], lattice_model.shape[1], 3), dtype=np.uint8)
        pygame.display.init()
        self.screen = pygame.display.set_mode([lattice_model.shape[0], lattice_model.shape[1]])
#        tick_time = clock.tick(fps)
        pygame.display.set_caption("Lattice gas")
        while True:
            self.update()

    def update(self):
        #print "update"
#        self.pixmap[0:20, :, :] = 255
        self.lattice_model.update()
        #print self.lattice_model.cells
        self.pixmap.fill(255)
        if self.lattice_model.lattice_type == 0:
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
        elif self.lattice_model.lattice_type == 1:
            self.pixmap[:, :, 0] -= (self.lattice_model.cells != 0b000000)*255
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100000)*128
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010000)*128
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001000)*128
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000100)*128
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000010)*128
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000001)*128

            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110000)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101000)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100100)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100010)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100001)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011000)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010100)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010010)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010001)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001100)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001010)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001001)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000110)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000101)*154
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000011)*154

            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111000)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110100)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110010)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110001)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101100)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101010)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101001)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100110)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100101)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100011)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011100)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011010)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011001)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010110)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010101)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010011)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001110)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001101)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001011)*179
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b000111)*179

            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111100)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111010)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111001)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110110)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110101)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110011)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101110)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101101)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101011)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b100111)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011110)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011101)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011011)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b010111)*205
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b001111)*205

            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111110)*230
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111101)*230
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111011)*230
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b110111)*230
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b101111)*230
            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b011111)*230

            # self.pixmap[:, :, 0] -= (self.lattice_model.cells == 0b111111)*255

#        scipy.ndimage.filters.gaussian_filter(self.pixmap[:, :, 0], 0.75, output=self.pixmap[:, :, 1])
        self.pixmap[:, :, 1] = self.pixmap[:, :, 0]
#        self.pixmap[:, :, 1] = self.pixmap[:, :, 0]
        self.pixmap[:, :, 2] = self.pixmap[:, :, 1]

        #print self.pixmap
        arr = np.asarray(self.pixmap)
        self.screen.fill((255, 255, 255))
        pygame.surfarray.blit_array(self.screen, arr)
        pygame.display.update()
        pygame.display.flip()

if __name__ == "__main__":
    print "main"
    
    node_types = pngnodes.read('600-400-pipe.png')
    height = node_types.shape[0]
    width = node_types.shape[1]

    density = 0.5

    model = LatticeModel(node_types, density, 1)

    model.cells[0:height*5/6, 0:width*5/6] = 0
#    model.cells[:, width*5/9:] = 0
    view = PygameView(model, 10)

