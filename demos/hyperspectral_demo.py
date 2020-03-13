import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
import sys


#from hypdatatools_algorithms import *
#from spectralinvariants import *

from hypdatatools_img import world2envi, plot_hyperspectral

datafolder = '/home/olli/Coding/hyperspectral'
hyperspectral_filename = 'subset_A_20170615_reflectance.hdr'

# open the data file -- reads only metadata
hypdata = spectral.open_image( os.path.join(datafolder, hyperspectral_filename) )

# open the file as a "memory map", a numpy matrix-like object. No need to worry about memory management.
hypdata_map = hypdata.open_memmap()

# the data in hypdata_map are a 3D array with coordinates [j,i,k]
# j = 'line' or the geographic inverted y-coordinate (from North to South)
# i = 'pixel' or the geographic x-coordinate (from West to East)
# k = 'band' or the spectroscopic coordinate, wavelength
# Note: As is mentioned above, the geographic coordinates x,y have inverted order in hyperspectral data 
#   and y-coordinate also inverted direction compared to common visualization. Pixel coordinates are 
#   counted from upper-left (North-West) corner of image. As elsewhere in python, indices start from zero.

print('Image size: %i by %i pixels, %i bands.' % (hypdata.shape[1], hypdata.shape[0], hypdata.shape[2]) )

# hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys() to get a list
# the key names originate (as plain text) from the .hdr file
# one of the keys is 'wavelength' (note: spectral converts key names by default to lowercase)

wavelength_stringlist = hypdata.metadata['wavelength'] # a list of strings is returned

# convert it to numpy array of floats
wavelength = np.array( wavelength_stringlist, dtype=float )

# choose bands for plotting based on their wavelength in nanometers
# note: in the data, wavelengths are given in micrometers
i_r =  abs(wavelength-0.660).argmin() # red band, the closest to 660 nm
i_g =  abs(wavelength-0.550).argmin() # green, closest band to 550 nm
i_b =  abs(wavelength-0.490).argmin() # blue, closest to 490 nm

# plot in true color
# imghandle = spectral.imshow(  hypdata_map, bands=[i_r,i_g,i_b] )
imghandle = plot_hyperspectral( os.path.join(datafolder,hyperspectral_filename), hypdata=hypdata, hypdata_map=hypdata_map, plotmode='RGB' )

# read in field-measured test plot data as a string matrix
plottable_filename = 'subset_A_pointids.txt' # 3 columns: ID, x, y in geographic coordinates
plottable = np.loadtxt( os.path.join( datafolder,plottable_filename), dtype='unicode', delimiter=',' )
plottable = np.loadtxt( os.path.join( datafolder,plottable_filename), dtype='S', delimiter=',' ).astype('str')

plotids = plottable[1:,0] # This returns column 0 as string
plotXY = plottable[1:,1:3].astype(float) # This returns columns 1,2 as floats
# NOTE: in python, to include column i, the range has to end with i+1. 

# convert from geographic to image coordinates
imageXY = world2envi(os.path.join(datafolder, hyperspectral_filename), plotXY )

# plot the test sites on the image with red crosses
imghandle.axes[0].plot( imageXY[:,0], imageXY[:,1], marker='x', color='red', linestyle=' ' )
# add plot IDs to the figure
for plotid,x,y in zip( plotids, imageXY[:,0], imageXY[:,1] ):
    imghandle.axes[0].annotate( plotid, (int(x+3.5),int(y-3.5) ), color='cyan' )

# plot a sample spectrum (of an arbitrary measurement plot) with matplotlib 
plt.figure() # open a new pyplot figure (a new window)
spectrumtoplot = 10 # choose 10th line in the table, for no particular reason
plt.plot( wavelength, hypdata_map[ int(round(imageXY[spectrumtoplot,1])), 
    int(round(imageXY[spectrumtoplot,0])), : ], label = "plot " + plotids[spectrumtoplot])
plt.plot( wavelength, hypdata_map[ int(round(imageXY[9,1])), 
    int(round(imageXY[9,0])), : ], label = "plot " + plotids[9])
plt.title("sample spectrum")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance factor * 10,000")
plt.legend() # make the legend visible
plt.show() # make the plot visible. needed in scripts to create the figure window
