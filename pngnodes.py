import png
import numpy
import itertools
from numpy import zeros

FREE_SPACE = 0
WALL = 1

def read(filename):
	f = open(filename, 'rb')

	reader = png.Reader(file=f)

	#returns the image boxed row, flat pixel, like [[R G B R G B], [R G B R G B]]
	#converts to RGB8 if neccessary
	(width, height, pixels, meta) = reader.asRGB8()

	#convert to numpy array so we can use slice indexing below
	imageRGB8 = numpy.vstack(itertools.imap(numpy.int64, pixels))

	f.close()

	nodes = numpy.zeros((height, width), dtype=int)

	

	for i in range(height):
		for j in range(width):
			pixel = 3*j+numpy.arange(3) #RGB triplet for pixel (i,j)
			if all(0 == imageRGB8[i,pixel]): #pixel is black
				nodes[j,i] = WALL
			else:
				nodes[j,i] = FREE_SPACE
			

	return nodes