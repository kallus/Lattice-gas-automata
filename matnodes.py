import numpy
import scipy.io
import pngnodes
import os

def read(filename):

	if not os.path.exists(filename):
		png_filename = filename.replace('.mat', '.png')
		print '.mat file not found. Reading {0} instead.'.format(png_filename)
		nodes = pngnodes.read(png_filename)
		d = dict(nodes=nodes)
		scipy.io.savemat(filename, d, oned_as='row')
		return nodes

	d = scipy.io.loadmat(filename)

	nodes = d['nodes'].copy()			

	return nodes