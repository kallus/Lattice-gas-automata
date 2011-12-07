import png
import numpy
import itertools
import sys
from numpy import zeros

SINK = -3
SOURCE = -2
WALL = -1

if __name__ == "__main__":
  if len(sys.argv) < 3:
    sys.exit("too few arguments")
  filename = sys.argv[2]
  figure = sys.argv[1]

  if figure == "pipe":
    W = 700
    H = 100
    domain = numpy.zeros((H,W), dtype=int)
    domain.fill(200)
    domain[0,:] = WALL
    domain[H-1,:] = WALL
    domain[H/6:3*H/6,W/2] = WALL
    domain[:,0] = SOURCE
    domain[:,W-1] = SINK
  elif figure == "porous":
    W = 700
    H = 100
    domain = numpy.zeros((H,W), dtype=int)
    domain.fill(200)
    domain[0,:] = WALL
    domain[H-1,:] = WALL
    randmat = numpy.random.rand(H-3, W/2)
    domain[1:H-2,W/4:3*W/4] = numpy.where(randmat < 0.1, numpy.ones(randmat.shape)*WALL, domain[1:H-2,W/4:3*W/4])
    domain[:,0] = SOURCE
    domain[:,W-1] = SINK
  elif figure == "porousfast":
    W = 700
    H = 100
    domain = numpy.zeros((H,W), dtype=int)
    domain.fill(200)
    domain[0,:] = WALL
    domain[H-1,:] = WALL
    randmat = numpy.random.rand(H-3, W/2)
    domain[1:H-2,W/4:3*W/4] = numpy.where(randmat < 0.01, numpy.ones(randmat.shape)*WALL, domain[1:H-2,W/4:3*W/4])
    domain[:,0] = SOURCE
    domain[:,W-1] = SINK

  picture = []
  domain = domain.T

  for i in xrange(H):
    row = []
    for j in xrange(W):
      x = domain[j, i]
      if x == SINK:
        row.append(255)
        row.append(0)
        row.append(0)
      elif x == SOURCE:
        row.append(0)
        row.append(255)
        row.append(0)
      elif x == WALL:
        row.append(0)
        row.append(0)
        row.append(255)
      else:
        row.append(x)
        row.append(x)
        row.append(x)
    picture.append(row)

  w = png.Writer(W, H)
  f = open(filename, "wb")
  w.write(f, picture)
  f.close()

