import png
import numpy
import itertools
from numpy import zeros

SINK = -3
SOURCE = -2
WALL = -1

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
			pixel = 3*j+numpy.arange(3) #indices for RGB triplet for pixel (i,j)
			rgb = imageRGB8[i, pixel]
			if all(abs(rgb - rgb[0]) < 20): #pixel is gray (or black or white)
				nodes[i,j] = 255 - rgb[0] #0 (black) is probability zero, 255 (white) is probability one
			elif all(rgb <= rgb[0]): #red
				nodes[i,j] = SINK
			elif all(rgb <= rgb[1]): #green
				nodes[i,j] = SOURCE
			elif all(rgb <= rgb[2]): #blue
				nodes[i,j] = WALL
			else:
				raise ValueError('Unexpected pixel value r={0}, g={1}, b={2}'.format(rgb[0], rgb[1], rgb[2]))



			

	return nodes