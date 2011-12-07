import os
import sys
import time
import numpy as np
import pygame
import pygame.display
import pygame.event
import scipy.ndimage
import png
from lattice_model import LatticeModel
import pngnodes

class PygameView(object):
    def __init__(self, lattice_model, delay, filename, lattice_type):
        print "pygame init"
        self.fps = 30
        self.lattice_model = lattice_model
        self.pixmap = np.ones((lattice_model.shape[0], lattice_model.shape[1], 3), dtype=np.uint8)
        self.transpose = np.zeros((lattice_model.shape[1], lattice_model.shape[0], 3), dtype=np.uint8)
        self.transpose2d = np.zeros((lattice_model.shape[1], lattice_model.shape[0]), dtype=np.uint32)
#        pygame.event.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode([lattice_model.shape[1], lattice_model.shape[0]]) #, pygame.HWSURFACE | pygame.FULLSCREEN)
#        tick_time = clock.tick(fps)
        pygame.display.set_caption(filename+' '+lattice_type)

        # self.wallmap = np.ones((lattice_model.shape[1], lattice_model.shape[0], 3), dtype=np.uint8)
        # self.wallmap.fill(0)
        # for i in xrange(lattice_model.shape[0]):
        #   for j in xrange(lattice_model.shape[1]):
        #     if lattice_model.node_types[i,j]==pngnodes.WALL:
        #       self.wallmap[j,i,0] = 12
        #       self.wallmap[j,i,1] = 82
        #       self.wallmap[j,i,2] = 232
        # self.wallsurface = pygame.surfarray.make_surface(self.wallmap)

        #self.wallsurface.set_colorkey((0, 0, 0))

        pygame.event.set_allowed(None)
        pygame.event.set_allowed(pygame.KEYDOWN)
        pygame.event.set_allowed(pygame.MOUSEBUTTONUP)

        self.lattice_model.update()
        self.transpose2d = self.lattice_model.cell_colors.T
        pygame.surfarray.blit_array(self.screen, self.transpose2d)
        pygame.display.flip()

        while (pygame.event.wait().type != pygame.KEYDOWN):
            pass

        pcounter = 0

        while True:
            events = pygame.event.get()
            for e in events:
                if pygame.KEYDOWN == e.type:
                    if e.unicode == u'p':
                        pcounter += 1
                        self.printscreen(filename+'.'+lattice_type+'.printscreen'+str(pcounter)+'.png')
                    elif e.unicode == u'q':
                        exit(1)
                elif pygame.MOUSEBUTTONUP == e.type:
                    pos = (e.pos[1], e.pos[0]) #e.pos is tuple (x, y)
                    if 1 == e.button:
                        self.lattice_model.add_particles(pos)
                    else:
                        self.lattice_model.remove_particles(pos)
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

#        scipy.ndimage.filters.gaussian_filter(self.lattice_model.cell_colors, 0.75,
#                                              output=self.lattice_model.cell_colors)
#        self.screen.fill((255, 255, 255))
        self.transpose2d = self.lattice_model.cell_colors.T
        pygame.surfarray.blit_array(self.screen, self.transpose2d)
        pygame.display.flip()

    def printscreen(self, filename):
        lattice_model = self.lattice_model
        picture = [None]*lattice_model.shape[0]
        for i in xrange(lattice_model.shape[0]):
            row = [None]*lattice_model.shape[1]*3
            c = 0
            for j in xrange(lattice_model.shape[1]):
                row[c] = lattice_model.cell_colors[i,j] >> 8 & 255
                c += 1
                row[c] = lattice_model.cell_colors[i,j] >> 16 & 255
                c += 1
                row[c] = lattice_model.cell_colors[i,j] >> 24 & 255
                c += 1
            picture[i] = row
        w = png.Writer(lattice_model.shape[1], lattice_model.shape[0])
        f = open(filename, "wb")
        w.write(f, picture)
        f.close()

if __name__ == "__main__":
    print "main"

    if len(sys.argv) != 3:
        print("Please specify lattice tpe and map file.")
        print("Usage: python pygame_view.py lattice_type file_name.png\n")
        print("lattice_type:\nhexagonal lattice: hex\nsquare lattice: square")
        exit(1)
    elif os.path.isfile(sys.argv[2]) == False:
        print("Map file name %s is not valid" % (sys.argv[1],))

    lattice_type = 1;
    if (sys.argv[1]=="square"):
        lattice_type = 0;
    
    node_types = pngnodes.read(sys.argv[2])
    filename = sys.argv[2]
    height = node_types.shape[0]
    width = node_types.shape[1]

    density = 0.5

    model = LatticeModel(node_types, density, lattice_type, 0)

#    model.cells[0:height*5/6, 0:width*5/6] = 0
#    model.cells[:, width*5/9:] = 0
    view = PygameView(model, 10, filename, sys.argv[1])

