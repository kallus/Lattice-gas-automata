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
			if all(rgb > 250): #user probably meant to paint white
				nodes[i,j] = 0 #probability zero
			elif all(abs(rgb - rgb[0]) < 50): #pixel is gray (or black or white)
				nodes[i,j] = 255 - rgb[0] #0 (black) is probability zero, 255 (white) is probability one
			elif all(rgb[0] - rgb[1:2] > 50): #significantly red
				nodes[i,j] = SINK
			elif rgb[1] - rgb[0] > 50 and rgb[1] - rgb[2] > 50: #significantly green
				nodes[i,j] = SOURCE
			elif all(rgb[2] - rgb[0:1] > 50): #significantly blue
				nodes[i,j] = WALL
			else:
				raise ValueError('Unexpected pixel value r={0}, g={1}, b={2}'.format(rgb[0], rgb[1], rgb[2]))



			

	return nodes