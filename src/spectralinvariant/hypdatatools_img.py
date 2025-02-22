"""
Copyright (C) 2017,2018  Matti Mõttus 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Some functions for working with hyperspectral data in ENVI file format
requires Spectral Python
the functions here do not depend on GDAL.
"""
import spectral
import spectral.io.envi as envi
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib
import numpy as np

from spectralinvariant.hypdatatools_utils import readtextfile

def get_wavelength(hypfilename, hypdata=None):
    """ Get the array of wavelengths in nm as numpy float array. If not present,
        return a range of decreasing integers starting at -1.
        
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
        
    Returns:
        wl : wavelength in nm
        wl_found : boolean flag
    """
    if hypdata is None:
        hypdata = spectral.open_image(hypfilename)

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata

    wl_found = 'wavelength' in hyp_metadata
    if wl_found:
        wl_hyp = np.array(hyp_metadata['wavelength'], dtype='float')
        if wl_hyp.max() < 100:
            # in microns, convert to nm
            wl_hyp *= 1000
    else:
        bands = int(hyp_metadata['bands'])
        wl_hyp = -1 * np.arange(bands) - 1  # start at -1
    return wl_hyp, wl_found
    
def get_DIV( hypfilename, hypdata=None ):
    """ read the Data Ignore Value from ENVI header, return None if not found
    
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.

    Returns:
    Data Ignore Value used in the data file
    """
    if hypdata is None:
        hypdata = spectral.open_image( hypfilename )

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata
        
    DIV_found = 'data ignore value' in hyp_metadata
    if DIV_found:
        DIV = hyp_metadata['data ignore value']
        # metadata contains strings. We need a number (float or int)
        if envi_isfloat( hypfilename, hyp_metadata ):
            DIV = float(DIV)
        else:
            DIV = int(DIV)
        return DIV
    else:
        return None
        
def get_scalefactor(hypfilename, hypdata=None, defaultSF=10000.0 ):
    """ read the Reflectance Scale Factor from ENVI header
    
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
    defaultSF: the common scale factor default value in hyperspectral images, 10,000
    
    Returns:
    the default scale factor value (i.e. defaultSF) if not found
    """
    if hypdata is None:
        hypdata = spectral.open_image( hypfilename )

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata
        
    try:
        scale_factor = float(hyp_metadata['reflectance scale factor'])
    except: # if scale factor missing from hyp metadata
        # next, check data type
        try:
            data_type = int( hypdata.metadata['data type'] )
        except: # if scale factor missing from hyp metadata
            return defaultSF
        if data_type in (1,2,3,12,13):
            # these are integer data codes. assume it's reflectance*10,000
            scale_factor = 10000.0

    return scale_factor
    
        
def get_pixelsize(hypfilename, hypdata=None ):
    """ Get the pixel size (in both x and y directions) of ENVI data from metadata in hypfilename 

    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
    
    Returns: (dx, dy)
    """
    
    if hypdata is None:
        hypdata = spectral.open_image( hypfilename )

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata

    if 'map info' in hyp_metadata:
        mapinfo = hyp_metadata['map info']
        dx = float(mapinfo[5])
        dy = float(mapinfo[6])
        pixelsize = [dx, dy]
    else:
        pixelsize = []
    return pixelsize


def get_imagesize(hypfilename, hypdata=None ):
    """ Get the image size of ENVI data file based on metadata in envihdrfilename 
    
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
        
    Returns: (x, y) [ x=samples, y=lines ]
    """
    if hypdata is None:
        hypdata = spectral.open_image( hypfilename )

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata

    if 'lines' in hyp_metadata and 'samples' in hyp_metadata:
        x = int(hyp_metadata['samples'])
        y = int(hyp_metadata['lines'])
        imagesize = [x, y]
    else:
        imagesize = []
    return imagesize


def get_geotrans(hypfilename, hypdata=None):
    """ Get the geometry transform of ENVI data file associated with hypfilename.
    Uses only hdr data, assumes that images are already oriented along the cardinal directions (i.e., no rotations)
    NOTE: this function ignores the start values in hdr files (as I am not sure how they work) 
    
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
        
    Returns: Geotransform (list with 6 numeric elements)
    """
    if hypdata is None:
        hypdata = spectral.open_image( hypfilename )

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata

    map_info = hyp_metadata['map info']  # the standard line containing image geometry description
    referencepixel = map_info[1:3]
    referencecoord = map_info[3:5]
    pixelsize = map_info[5:7]
    # convert these lists of strings to int or float using list comprehension
    referencepixel = [int(float(x)) for x in referencepixel]
    referencecoord = [float(x) for x in referencecoord]
    pixelsize = [float(x) for x in pixelsize]

    # What we need are GeoTrans (GT) coefficients, for transforming between
    #    pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space
    # Xp = GT[0] + P*GT[1] + L*GT[2];
    # Yp = GT[3] + P*GT[4] + L*GT[5];
    # The inverse in the general case is  
    # P = ( Xp*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - Yp*GT[2] ) / ( GT[1]*GT[5] - GT[2]*GT[4] )
    # L = ( Yp*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - Xp*GT[4] ) / ( GT[1]*GT[5] - GT[2]*GT[4] )
    # NOTE: ENVI files refer to pixels by their upper-left corner. It is more convenient to use pixel center coordinates
    #   if the coordinates p and l are given relative to pixel centers, we get
    # Xp = GT[0] + (p+0.5)*GT[1] + (l+0.5)*GT[2];
    # Yp = GT[3] + (p+0.5)*GT[4] + (l+0.5)*GT[5];
    # D = GT[1]*GT[5] - GT[2]*GT[4]
    # p = ( Xp*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - Yp*GT[2] ) / D - 0.5
    # l = ( Yp*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - Xp*GT[4] ) / D - 0.5

    GT = np.zeros(6)
    GT[0] = referencecoord[0] - (referencepixel[0] - 1) * pixelsize[0]  # x-coordinate of upper-left pixel
    GT[1] = pixelsize[0]
    GT[3] = referencecoord[1] + (referencepixel[1] - 1) * pixelsize[
        1]  # y-coordinate of upper-left pixel. Note: y-axis is inverted
    GT[5] = -pixelsize[1]

    return GT

def avg_spectrum(hypfilename, coordlist, DIV=-1, hypdata=None, hypdata_map=None):
    """ exctract data point values from hypfilename
    
    Args:
        hypfilename: name of hyperspectral data file
        coordlist: list of two lists: [ [y] , [x] ] in image coordinates
            NOTE! Envi BIL files have y (line) for first coordinate [0], x (pixel) for second [1]
        DIV : value used as Data Ignore Value. If not used, set DIV=-1
        hypdata, hypdata_map: spectral handles for file and memmap (optional). If given, hypfilename will not be reopened
        
    Returns:
        avg_spectrum: average spectrum for all points
        N: number of spectra averaged
    """
    if hypdata is None:
        # open the file if not open yet. This only gives access to metadata.                
        hypdata = spectral.open_image(hypfilename)
        # open the file as memmap to get the actual hyperspectral data
        hypdata_map = hypdata.open_memmap()  # open as BIP

    hypdata_sub = hypdata_map[coordlist[0],coordlist[1],:]
    if DIV != -1:
        # look for no data values
        hypdata_sub_min = np.min(hypdata_sub, axis=1)
        hypdata_sub_max = np.max(hypdata_sub, axis=1)
        sub_incl = np.where(np.logical_and(hypdata_sub_min != float('nan'),
                                           np.logical_or(hypdata_sub_min != DIV, hypdata_sub_max != DIV)))[0]
        hypdata_sub = hypdata_sub[sub_incl, :]
    N = (hypdata_sub.shape[0])
    spectrum = DIV*np.ones(hypdata_map.shape[2]) if N==0 else np.mean(hypdata_sub, axis=0)
    return spectrum, N


def envi_isfloat( hypfilename, hypdata=None ):
    """ return if the file type contains floating-point.
    If False, it's integer or byte. If type not given, return None. From documentation:
        'data type' The type of data representation, where 1=8-bit byte; 2=16-bit
        signed integer; 3=32-bit signed long integer; 4=32-bit floating
        point; 5=64-bit double-precision floating point; 6=2x32-bit
        complex, real-imaginary pair of double precision; 9=2x64-bit
        double-precision complex, real-imaginary pair of double precision;
        12=16-bit unsigned integer; 13=32-bit unsigned long integer;
        14=64-bit signed long integer; and 15=64-bit unsigned long
        integer.
        
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
    """
    if hypdata is None:
        hypdata = spectral.open_image( hypfilename )

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata
        
    type_found = 'data type' in hyp_metadata
    if type_found:
        dt = int(hyp_metadata['data type'])
        if dt>3 and dt<12:
            return True
        else:
            return False
    else:
        # data type not given. Output undetermined
        return None
        
def plot_hyperspectral( hypfilename, hypdata=None, hypdata_map=None, outputcommand=None, 
    plotmode='default', plotbands=None, fig_hypdata=None, clip_up=0.95, clip_upvalue=None,
    clip_low=None, clip_lowvalue=0, stretch_individual=True ):
    """ Create a figure with hyperspectral image and return handle
       before using, check if SPy.imshow() could be used instead
    
    Args:
    hypfilename: the name + full path to Envi hdr file
    hypdata is a Spectral Python file handle. If set, hyperspectral file will not be reopened.
        alternatively, it can be the metadata dictionary.
    hypdata_map: 3D np.ndarray-like object, e.g. the memory map of hypdata file handle
    outputdata is the optional print command for redirecting output
    plotmode = 'RGB', 'NIR', 'default', 'falsecolor'. Which mode to use for plotting. Overriden by plotbands (if set).
        default means looking for the suggested bands in hdr, if not found, use RGB
        falsecolor: plot with band #0 as hue. Useful for classified images. 
            was slow because of hsv->rgb conversion, but not anymore as is done by imshow()
    plotbands = list [r,g,b], if not set, guessed from metadata and elsewhere
    fig_hypdata: figure handle to use and return
    clip_up: the percentile above which to draw as white, ignored if clip_upvalue is set
    clip_upvalue: the value above which to draw as white
    clip_low: the percentile below which everything is black, ignored by default (unless set)
    clip_lowvalue: the value below which everything is black, ignored if clip_low set
    stretch_individual: whether to stretch bands individually
    """

    functionname = "plot_hyperspectral(): "  # used in messaging

    if hypdata is None:
        hypdata = spectral.open_image(hypfilename)
        hypdata_map = hypdata.open_memmap()

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata

    filename_short = os.path.split(hypfilename)[1]
    filename_short = os.path.splitext(filename_short)[0]

    if outputcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        outputcommand = lambda x: print(x,end='',flush=True)
    
    hypdata_map_shape = ",".join( map(str,hypdata_map.shape) )
    outputcommand( functionname + filename_short + " dimensions " + hypdata_map_shape+". \n") #shape[0]==lines, shape[1]==pixels, shape[2]==bands

    plot_rgb = True # flag: plot with three bands (as opposed to monochrome)
    falsecolor = False # flag: plot as falsecolor pallette using the first band
    
    if plotbands is None:
        if plotmode == 'default':
            if 'default bands' in hyp_metadata:
                if len( hyp_metadata['default bands'] ) > 2:
                    i_r = int( float(hyp_metadata['default bands'][0]) ) - 1
                    # note: direct conversion from string to int does not work if string contains decimals (e.g., '97.0')
                    i_g = int( float(hyp_metadata['default bands'][1]) ) - 1
                    i_b = int( float(hyp_metadata['default bands'][2]) ) - 1
                    # avoid official printing band names, they usually contain long crappy strings
                    outputcommand(
                        functionname + filename_short + ": using default bands (%i,%i,%i) for plotting RGB. \n" % (
                        i_r, i_g, i_b))
                else:
                    plotmode = 'RGB'
            else:
                plotmode = 'RGB'

        # wavelengths should be in metadata
        # these will be stored in the class for other functions to use (interpolation and plotting of reference data)
        wl_hyp, wl_found = get_wavelength(hypfilename, hypdata)
        if wl_found:
            if plotmode == 'NIR':
                i_r = abs(wl_hyp - 780).argmin()  # red band, use NIR channel
                i_g = abs(wl_hyp - 670).argmin()  # green band, use red channel
                i_b = abs(wl_hyp - 550).argmin()  # blue band, use green channel
            else:
                # plotmode should be 'RGB', but make this also the default case
                i_r = abs(wl_hyp - 670).argmin()  # red band
                i_g = abs(wl_hyp - 550).argmin()  # green
                i_b = abs(wl_hyp - 450).argmin()  # blue
            outputcommand(functionname + filename_short + ": using wavelengths for plotting, ")
            outputcommand("%5.1f,%5.1f,%5.1f nm.\n" % (wl_hyp[i_r], wl_hyp[i_g], wl_hyp[i_b]))
        else:
            # just use the first one or three bands
            if hypdata_map.shape[2] > 2:
                # we have at least 3 bands
                i_r = 0
                i_g = 1
                i_b = 2
                if 'band names' in hyp_metadata:
                    name_r = hyp_metadata['band names'][i_r]
                    name_g = hyp_metadata['band names'][i_g]
                    name_b = hyp_metadata['band names'][i_b]
                else:
                    name_r = 'band ' + str(i_r)
                    name_g = 'band ' + str(i_g)
                    name_b = 'band ' + str(i_b)
                outputcommand(
                    functionname + filename_short + " display bands: " + name_r + ", " + name_g + ", " + name_b + ".\n")
            else:
                # monochromatic, use first band only
                i_r = 0
                if 'band names' in hyp_metadata:
                    name_r = hyp_metadata['band names'][i_r]
                else:
                    name_r = 'band0'
                plot_rgb = False
                outputcommand(functionname + "Plotting " + filename_short + " as monochrome with band0.\n")
    else: # if plotbands is None:
        # outputcommand( functionname + "Bands to plot given as argument, "+str(len(plotbands))+" band(s).\n")
        if len(plotbands) >= 3:
            plot_rgb = True
            i_r = plotbands[0]
            i_g = plotbands[1]
            i_b = plotbands[2]
        else:
            plot_rgb = False
            i_r = plotbands[0]
            
    if plotmode == 'falsecolor':
        # monochromatic, use first band only as the hue component
        # outputcommand(functionname + "Plotting " + filename_short + " as falsecolor.\n")
        if 'band names' in hyp_metadata:
            name_r = hyp_metadata['band names'][i_r]
        else:
            name_r = 'band0'
        plot_rgb = False
        falsecolor = True

    # set max area
    # plotsize_max = 1024
    # plotsize_max = 5000
    # plot_pixmax = min(plotsize_max,hypdata_map.shape[1])
    # plot_rowmax = min(plotsize_max,hypdata_map.shape[0])

    # plot using pyplot.imshow -- this allows to catch clicks in the window
    outputcommand(functionname + "reading data...")
    if type(hypdata) is dict:
        if plot_rgb:
            hypdata_rgb = hypdata_map[:, :, (i_r, i_g, i_b)].astype('float32')
        else:
            hypdata_rgb = hypdata_map[:, :, i_r].astype('float32')
    else:
        if plot_rgb:
            hypdata_rgb = hypdata.read_bands( (i_r, i_g, i_b) ).astype('float32')
        else:
            hypdata_rgb = hypdata.read_bands([i_r]).astype('float32')
    if plot_rgb:
        fig_hypdata = plot_hypdatamatrix( hypdata_rgb, plottitle=filename_short,
            fig_hypdata=fig_hypdata, clip_up=clip_up, clip_upvalue=clip_upvalue, 
            clip_low=clip_low, clip_lowvalue=clip_lowvalue, stretch_individual=stretch_individual,
            outputcommand=outputcommand )
    else: 
        fig_hypdata = plot_hypdatamatrix_singleband( hypdata_rgb, plottitle=filename_short,
            falsecolor=falsecolor, fig_hypdata=fig_hypdata, clip_up=clip_up, clip_upvalue=clip_upvalue, 
            clip_low=clip_low, clip_lowvalue=clip_lowvalue, stretch_individual=stretch_individual,
            outputcommand=outputcommand )

    return fig_hypdata


def plot_singleband(hypfilename, hypdata=None, hypdata_map=None, bandnumber=None, fig_hypdata=None,
    clip_up=0.95, clip_upvalue=None, clip_low=None, clip_lowvalue=0,
    outputcommand=None):
    """ Create a figure with a single band from hypersectral image and return handle
        before using, check if SPy.imshow() could be used instead
    
    Args:
    hypfilename: the name + full path to Envi hdr file
    hypdata: a Spectral Python file handle. If set, hyperspectral file will not be reopened.
        alternatively, it can be the metadata dictionary.
    hypdata_map: 3D np.ndarray-like object, e.g. the memory map of hypdata file handle
    fig_hypdata: figure handle to use and return
    clip_up: the percentile above which to draw as white, ignored if clip_upvalue is set
    clip_upvalue: the value above which to draw as white
    clip_low: the percentile below which everything is black, ignored by default (unless set)
    clip_lowvalue: the value below which everything is black, ignored if clip_low set
    outputcommand: the optional print command for redirecting output
    returns figure handle
    """
    functionname = "plot_singleband(): "  # used in messaging
    if hypdata is None:
        hypdata = spectral.open_image(hypfilename)
        hypdata_map = hypdata.open_memmap()

    if type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        hyp_metadata = hypdata.metadata

    if outputcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        outputcommand = lambda x: print(x,end='',flush=True)
        
    filename_short = os.path.split(hypfilename)[1]
    filename_short = os.path.splitext(filename_short)[0]
    
    hypdata_map_shape = ",".join( map(str,hypdata_map.shape) )

    # try to use the default band. if more than one given, use the first. Otherwise, use first band.
    if bandnumber == None:
        if 'default bands' in hypdata.metadata:
            bandnumber = int(hyp_metadata['default bands'][0]) - 1
            # avoid official printing band names, they usually contain long crappy strings
            outputcommand( functionname 
                 + filename_short + ": using the default band %i for plotting RGB. \n" % bandnumber)
        else:
            bandnumber = 0

    # plot using pyplot.imshow -- this allows to catch clicks in the window
    if type(hypdata) is dict:
        hypdata_i = np.squeeze(hypdata_map[:, :, bandnumber]).astype('float32')
    else:
        hypdata_i = np.squeeze(hypdata.read_bands([bandnumber]).astype('float32'))

    outputcommand( functionname + filename_short + " dimensions " + hypdata_map_shape+
        " band " + str(bandnumber) + " min " + str(np.nanmin(hypdata_i)) + " max " + str(np.nanmax(hypdata_i)) + "\n") #shape[0]==lines, shape[1]==pixels, shape[2]==bands
    
    # the actual work in a separate function
    plot_hypdatamatrix_singleband( hypdata_i, plottitle=filename_short,
        falsecolor=False, fig_hypdata=fig_hypdata, clip_up=clip_up, clip_upvalue=clip_upvalue, 
        clip_low=clip_low, clip_lowvalue=clip_lowvalue, outputcommand=outputcommand )

    return fig_hypdata


def plot_hypdatamatrix( hypdata_rgb, plottitle="", fig_hypdata=None,
    clip_up=0.95, clip_upvalue=None, clip_low=None, clip_lowvalue=0, stretch_individual=True,
    outputcommand=None):
    """ Plot a 2D matrix using imshow() as RGB or grayscale
        the actual plotting work for plotting a RGB hyperspectral data matrix
            before using, check if SPy.imshow() could be used instead
            
    Args:
        hypdata_rgb: 3-band numpy matrix
        plottitle: string with the plot title
        fig_hypdata: figure handle to use and return
        clip_up: the percentile above which to draw as white, ignored if clip_upvalue is set
        clip_upvalue: the value above which to draw as white
        clip_low: the percentile below which everything is black, ignored by default (unless set)
        clip_lowvalue: the value below which everything is black, ignored if clip_low set
        stretch_individual: whether to stretch bands individually

    Returns: figure handle
    """
    
    axestoapply = None; # this will apply the command along all axis (whole datacube)
    if stretch_individual:
        axestoapply = (0,1) # apply along x,y, i.e., separately for each band
        
    functionname = "plot_hypdatamatrix(): "  # used in messaging

    if outputcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        outputcommand = lambda x: print(x,end='',flush=True)
    outputcommand(" calculating range...")
    # make a copy not to modify the original rgb
    hypdata_rgb_plot = hypdata_rgb.copy()
    
    # stretch the image: calculate nice scaling if needed
    outputcommand(" calculating scaling...")
    N_notnan = None # it will be computed only when needed
    if clip_low is not None:
        # scale to the given lowe percentile. Add it to data if it's negative
        # add their most negative value to bands which have negative values 
        N_notnan = np.array( np.count_nonzero( ~np.isnan(hypdata_rgb_plot), axis=axestoapply ) ).min() 
        if N_notnan < 200:
            outputcommand("\n" + functionname + "Too few ("+str(N_notnan)
                +") pixels with values, not applying histogram.\n")
            datamin = np.nanmin( hypdata_rgb_plot, axis=axestoapply )
            datamax = np.nanmax( hypdata_rgb_plot, axis=axestoapply )
            clip_lowvalue = datamin + clip_low*(datamax-datamin)
        elif ( np.array( np.nanmax(hypdata_rgb_plot, axis=axestoapply) - np.nanmin(hypdata_rgb_plot, axis=axestoapply) ) ).min() > 0:
            clip_lowvalue = np.nanpercentile(hypdata_rgb_plot, clip_low*100, axis=axestoapply )
        else:
            outputcommand("\n" + functionname + "All pixels have the same value, no scaling.\n")
            clip_lowvalue = 0 

    if clip_upvalue is None:
        if N_notnan is None:
            # not computed yet, compute!
            N_notnan = np.array( np.count_nonzero( ~np.isnan(hypdata_rgb_plot), axis=axestoapply ) ).min() 
        if N_notnan < 200:
            outputcommand("\n" + functionname + "Too few ("+str(N_notnan)
                +") pixels with values, not applying histogram.\n")
            datamin = np.nanmin(hypdata_rgb_plot, axis=axestoapply )
            datamax = np.nanmax(hypdata_rgb_plot, axis=axestoapply )
            datascaling = datamin + clip_up*(datamax-datamin)
        elif np.array( np.nanmax(hypdata_rgb_plot, axis=axestoapply) - np.nanmin(hypdata_rgb_plot, axis=axestoapply) ).min()  > 0:
            datascaling = np.nanpercentile(hypdata_rgb_plot, clip_up*100, axis=axestoapply )
        else:
            outputcommand("\n" + functionname + "All pixels have the same value, no scaling.\n")
            datascaling = 1.0 
    else:
        datascaling = clip_upvalue
        
    outputcommand(" scaling between " + str(clip_lowvalue) + " and " + str(datascaling) + "...")
    hypdata_rgb_plot = (hypdata_rgb_plot-clip_lowvalue) / (datascaling-clip_lowvalue)

    # scaling alone seems to give not so nice plots
    hypdata_rgb_plot[hypdata_rgb_plot > 1] = 1
    hypdata_rgb_plot[hypdata_rgb_plot < 0] = 0

    if fig_hypdata is None:
        fig_hypdata = plt.figure()  # create a new figure
        outputcommand( " creating new figure...")
    else:
        fig_hypdata.clf() # clear the figure
        outputcommand( " reusing old figure...")
        
    ax0 = fig_hypdata.add_subplot(1, 1, 1)

    outputcommand(" displaying...")
    ax0.imshow(hypdata_rgb_plot)  # ax0 is fig_hypdata.axes[0]
    ax0.set_title( plottitle )

    fig_hypdata.canvas.draw()
    fig_hypdata.show()
    outputcommand(" done.\n")
    return fig_hypdata
    
def plot_hypdatamatrix_singleband( hypdata_band, plottitle="", falsecolor=False, fig_hypdata=None,
    clip_up=0.95, clip_upvalue=None, clip_low=None, clip_lowvalue=0,
    outputcommand=None ):
    """ Plot a 2D matrix using imshow()
        the actual plotting work for plotting a single-band hyperspectral data matrix
            before using, check if SPy.imshow() could be used instead
            
    Args:
        hypdata_band: 1-band numpy matrix
        plottitle: string with the plot title
        falsecolor: plot with band #0 as hue. Useful for classified images. SLOW because of hsv->rgb conversion
        fig_hypdata: figure handle to use and return
        clip_up: the percentile above which to draw as white, ignored if clip_upvalue is set
        clip_upvalue: the value above which to draw as white
        clip_low: the percentile below which everything is black, ignored by default (unless set)
        clip_lowvalue: the value below which everything is black, ignored if clip_low set
        
    Returns: returns figure handle
    """

    functionname = "plot_hypdatamatrix_singleband(): "  # used in messaging

    if outputcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        outputcommand = lambda x: print(x,end='',flush=True)
    
    # we want a 2-d matrix
    if hypdata_band.ndim > 2:
        hypdata_band = np.squeeze( hypdata_band )
        if hypdata_band.ndim > 2:
            hypdata_band = hypdata_band[:,:,0]

    # stretch the image: calculate nice scaling if needed
    outputcommand(functionname + " calculating scaling...")
    N_notnan = None # it will be computed only when needed
    if clip_low is not None:
        # scale to the given lowe percentile. Add it to data if it's negative
        # add their most negative value to bands which have negative values 
        N_notnan = np.count_nonzero( ~np.isnan(hypdata_band) )
        if N_notnan < 200:
            outputcommand("\nToo few ("+str(N_notnan)
                +") pixels with values, not applying histogram.\n")
            datamin = np.nanmin(hypdata_band)
            datamax = np.nanmax(hypdata_band)
            clip_lowvalue = datamin + clip_low*(datamax-datamin)
        elif ( np.nanmax(hypdata_band) - np.nanmin(hypdata_band) ) > 0:
            clip_lowvalue = np.nanpercentile(hypdata_band, clip_low*100 )
        else:
            outputcommand("\nAll pixels have the same value, no scaling.\n")
            clip_lowvalue = 0 
            
    if clip_upvalue is None:
        ii = hypdata_band > 0
        if np.where(ii)[0].size < 200:
            datascaling = hypdata_band.max()
            datascaling /= clip_up
            outputcommand("\nToo few pixels with values, not applying histogram.\n")
        elif np.ptp(hypdata_band[ii]) > 0:
            datascaling = np.percentile(hypdata_band[ii], clip_up*100 )
        else:
            outputcommand("\n" + functionname + "All pixels are the same, no scaling.\n")
            datascaling = 1.0 
    else:
        datascaling = clip_upvalue

    outputcommand(" scaling between " + str(clip_lowvalue) + " and " + str(datascaling) + "...")
    hypdata_plot = (hypdata_band-clip_lowvalue) / (datascaling-clip_lowvalue)
    # this also forces the hypdata_plot to be a copy -- hypdata_band may be read-only

    # percentile alone seems to give not so nice plots
    hypdata_plot[hypdata_plot > 1] = 1
    hypdata_plot[hypdata_plot < 0] = 0

    if fig_hypdata is None:
        fig_hypdata = plt.figure()  # create a new figure
        outputcommand( " creating new figure...")
    else:
        fig_hypdata.clf() # clear the figure
        outputcommand( " reusing old figure...")

    ax0 = fig_hypdata.add_subplot(1, 1, 1)   
#    if falsecolor:
#        outputcommand(" ... converting from HSV (SLOW!!!)...")
#        hypdata_hsv = np.stack([hypdata_plot, np.ones_like(hypdata_plot), np.ones_like(hypdata_plot)], axis=-1)
#        hypdata_plot = matplotlib.colors.hsv_to_rgb(hypdata_hsv)
    outputcommand(" displaying...")
    if falsecolor:
        ax0.imshow(hypdata_plot)  # ax0 is fig_hypdata.axes[0]
    else:
        ax0.imshow(hypdata_plot,cmap='gray', vmin=0, vmax=1)  # ax0 is fig_hypdata.axes[0]
    ax0.set_title( plottitle )

    fig_hypdata.canvas.draw()
    fig_hypdata.show()
    outputcommand(" done.\n")
    return fig_hypdata
    

def set_display_square(windowhandle):
    """
    set the displayed area to be a square by extending image in the shorter direction
    windowhandle: pyplot figure handle
    """
    xlim = windowhandle.axes[0].get_xlim()
    ylim = windowhandle.axes[0].get_ylim()
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    xcenter = (xlim[1] + xlim[0]) / 2
    ycenter = (ylim[1] + ylim[0]) / 2
    range_out = max(np.abs(xrange), np.abs(yrange))
    xrange_out = xrange / (np.abs(xrange) / range_out)
    yrange_out = yrange / (np.abs(yrange) / range_out)
    minx_out = int(xcenter - xrange_out / 2)
    maxx_out = int(xcenter + xrange_out / 2)
    miny_out = int(ycenter - yrange_out / 2)
    maxy_out = int(ycenter + yrange_out / 2)

    windowhandle.axes[0].set_xlim((minx_out, maxx_out))
    windowhandle.axes[0].set_ylim((miny_out, maxy_out))
    windowhandle.canvas.draw()


def zoomtoimage(fig_hypdata, hypdata_map):
    """
    Try to zoom fig_hypdata (a matpotlib figure handle) to extents of hypdata_map (a raster or similar)
    if hypdata_map is None, use simple autoscale(), which does the work sometimes, especially if image is the only thing in the figure
    
    Args:
    hypfilename: name of the hyperspectral ENVI file (header file .hdr). Not used if hypdata given
    hypdata: a Spectral Python file handle or the hyperspectral metadata dictionary.
        If set, hyperspectral data file will not be reopened.
    """
    if hypdata_map is None:
        fig_hypdata.axes[0].autoscale()  # not always working??
    else:
        maxx = hypdata_map.shape[1]
        maxy = hypdata_map.shape[0]
        fig_hypdata.axes[0].set_xlim((-0.5, maxx - 0.5))
        fig_hypdata.axes[0].set_ylim((maxy - 0.5, -0.5))
        fig_hypdata.canvas.draw()


def create_raster_like(envifile, outfilename, Nlayers=1, outtype=4, interleave='bsq', fill_black=False, force=True,
                       description=None, metadata_keys_copy=[], metadata_add=None, localprintcommand=None):
    """ Create a new envi raster of the same size and geometry as the input file (envifile)
    
    Args:
        envifile: the ENVI (Spectral python) raster to use as the example
        outfilename: the filename to create
        Nlayers: number of layers in the output file
        outtype: output file data type according to ENVI
            ENVI data types
                1=8-bit byte;
                2=16-bit signed integer;
                3=32-bit signed long integer;
                4=32-bit floating point; 
                5=64-bit double-precision floating point; 
                6=2x32-bit complex, real-imaginary pair of double precision;
                9=2x64-bit double-precision complex, real-imaginary pair of double precision;
                12=16-bit unsigned integer;
                13=32-bit unsigned long integer;
                14=64-bit signed long integer; 
                15=64-bit unsigned long integer.
        interleave: 'bil', 'bip, or 'bsq' (default bsq)
        fill_black: whether to fill the new file with zeros (DIVs) 
        force: whether to overwrite existing file
        description: what to write in the description header field
        metadata_keys_copy: which metadata keys to copy from the metadata of the original image in addition to the default keys
            see code for default keys, originally they included ['map info', 'coordinate system string', 'sensor type']
        metadata_add: new metadata to be added at file creation, must be a dict

    Returns:
        outfilehandle. To write to that raster, use outdata_map = outdata.open_memmap( writable=True )
    """

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)

    functionname = "create_raster_like():"

    # make sure the extension is .hdr
    ofparts = os.path.splitext(outfilename)
    outfilename = ofparts[0] + '.hdr'

    metadata = {}  # start from scratch with the new metadata dictionary

    if type(envifile) is str:
        # assume it is a filename, open it
        hypdata = spectral.open_image(envifile)
    else:
        # assume envifile is a Spectral Python file handle
        hypdata = envifile

    default_metadata_keys = ['map info', 'coordinate system string', 'sensor type']
    
    for key in default_metadata_keys + metadata_keys_copy:
        key_lower = str(key).lower()
        # assume that SPy converts the keys it reads to lowercase automatically
        if key_lower in hypdata.metadata.keys():
            metadata[key_lower] = hypdata.metadata[key_lower]

    metadata['lines'] = hypdata.nrows
    metadata['samples'] = hypdata.ncols
    metadata['bands'] = Nlayers
    metadata['data type'] = outtype
        
    metadata['data ignore value'] = "0"

    if description is None:
        description = 'A new raster based on the geometry of ' + hypdata.filename
    if 'description' in metadata:
        description = description + ' : ' + metadata['description']
        
    metadata['description'] = description
    metadata['interleave'] = interleave
    
    if type(metadata_add) is dict:
        metadata.update( metadata_add )

    localprintcommand(functionname + " creating file " + outfilename)

    outdata = envi.create_image(outfilename, metadata=metadata, interleave=interleave, ext='', force=force)
    outdata_map = outdata.open_memmap(writable=True)

    if fill_black:
        localprintcommand(", filling with DIV ") 
        DIV = get_DIV( envifile )
        if DIV is None:
            localprintcommand(" (no DIV in example, setting to 0) ... ") 
            DIV = 0
        else:
            localprintcommand("... ")
        outdata_map[:,:,:] = DIV
    else:
        localprintcommand("... ")

    localprintcommand(" done .\n") # the file will be closed as the function exits

    return outdata


def subset_raster(hypdata, outfilename, subset, hypdata_map=None, interleave=None, localprintcommand=None):
    """ subset the raster by local image coordinates (starting with 0,0)
    
    Args:
        hypdata = the ENVI (Spectral python) raster to subset. Note: if hypdata_map is given, hypdata is not reopened
        subset = integers [ xmin, ymin, xmax, ymax ]
            if subset is larger than image, subset is shrunk
            the included range is min:max (i.e., max itself will be excluded)
        hypdata_map = the raster handle of the input file. If given, hypdata is not reopened
    interleave: 'bil', 'bip, or 'bsq'
        if None, the interleave of input image is used
    """
    if hypdata_map is None:
        hypdata_map = hypdata.open_memmap()

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)

    functionname = "subset_raster():"

    imaxx = hypdata_map.shape[1]
    imaxy = hypdata_map.shape[0]

    xmin = int(subset[0])
    ymin = int(subset[1])
    xmax = int(subset[2])
    ymax = int(subset[3])

    if xmin >= xmax or ymin >= ymax:
        localprintcommand(functionname + " No pixels in the subset area. Aborting.\n")
    else:
        if xmin < 0 or xmax > imaxx or ymin < 0 or ymax > imaxy:
            # there will be at least one black border to fill in with DIV
            fill_black = True
        else:
            fill_black = False

        # make sure the extension is .hdr
        ofparts = os.path.splitext(outfilename)
        outfilename = ofparts[0] + '.hdr'
        metadata = copy.copy(hypdata.metadata)  # maybe copy.deepcopy should be better?
        mapinfo = metadata['map info']
        # shift the starting coordinate
        x0 = float(mapinfo[3])
        y0 = float(mapinfo[4])
        dx = float(mapinfo[5])
        dy = float(mapinfo[6])
        x0_out = x0 + xmin * dx
        y0_out = y0 - ymin * dy
        mapinfo[3] = str(x0_out)
        mapinfo[4] = str(y0_out)

        metadata['map info'] = mapinfo
        metadata['lines'] = str(ymax - ymin)
        metadata['samples'] = str(xmax - xmin)

        if "data ignore value" in metadata:
            DIV = metadata['data ignore value']
        else:
            localprintcommand(functionname + " setting output file Data Ignore Value to 0.\n")
            metadata['data ignore value'] = "0"
            DIV = 0

        if 'description' in metadata:
            description = metadata['description']
        else:
            description = ''
        description = 'spatial subset of : ' + description
        metadata['description'] = description

        if interleave is None:
            interleave = hypdata.metadata['interleave']
        else:
            hypdata.metadata['interleave'] = interleave

        # the indices to subset the original data
        #shrink the subset if needed
        imin = max(xmin, 0)
        imax = min(xmax, imaxx) 
        jmin = max(ymin, 0)
        jmax = min(ymax, imaxy) 
        # offset in new data
        if xmin < 0:
            i0 = -xmin
        else:
            i0 = 0
        if ymin < 0:
            j0 = -ymin
        else:
            j0 = 0
        # [i0,j0] is the location of the [0,0] pixel of original image in the subset image

        localprintcommand(functionname + " new image size: requested %ix%i, final %ix%i, offset %i,%i.\n" %
                          (xmax - xmin, ymax - ymin, imax - imin, jmax - jmin, i0, j0))
        localprintcommand(functionname + " creating file " + outfilename)
        outdata = envi.create_image(outfilename, metadata=metadata, interleave=interleave, ext='', force=True )
        localprintcommand(" and saving data ... ")
        outdata_map = outdata.open_memmap(writable=True)
        if fill_black:
            outdata_map[:, :, :] = DIV
        outdata_map[j0:j0 + jmax - jmin, i0:i0 + imax - imin, :] = hypdata_map[jmin:jmax, imin:imax, :]
        localprintcommand(" done .\n")  # actually, the file will be closed after function exits


def crop_raster(hypdata, outfilename, subset, hypdata_map=None, interleave=None, localprintcommand=None):
    """
    subset the raster by local image coordinates (starting with 0,0)
    subset = integers [ xmin, ymin, xmax, ymax ]
       ONLY CROP -- xmin and ymin have to larger than zero, and xmax and ymax smaller than image dimensions
       to allow for black pixels, use subset_raster()
       the included range is min:max (i.e., max itself will be excluded)
    interleave: 'bil', 'bip, or 'bsq'
        if None, the interleave of input image is used
    """

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)
        
    if hypdata_map is None:
        hypdata_map = hypdata.open_memmap()

    functionname = "crop_raster():"

    imaxx = hypdata_map.shape[1]
    imaxy = hypdata_map.shape[0]

    xmin = int(subset[0])
    ymin = int(subset[1])
    xmax = int(subset[2])
    ymax = int(subset[3])

    # simple sanity checks
    xmin = max(xmin, 0)
    xmax = min(xmax, imaxx)
    ymin = max(ymin, 0)
    ymax = min(ymax, imaxy)

    if xmin >= xmax or ymin >= ymax:
        localprintcommand(functionname + " No pixels in the subset area. Aborting.\n")
    else:
        # make sure the extension is .hdr
        ofparts = os.path.splitext(outfilename)
        outfilename = ofparts[0] + '.hdr'
        metadata = hypdata.metadata
        mapinfo = metadata['map info']
        if 'description' in metadata:
            description = metadata['description']
        else:
            description = ''
        description = 'spatial subset of : ' + description
        metadata['description'] = description
        # shift the starting coordinate
        x0 = float(mapinfo[3])
        y0 = float(mapinfo[4])
        dx = float(mapinfo[5])
        dy = float(mapinfo[6])
        x0_out = x0 + xmin * dx
        y0_out = y0 - ymin * dy
        mapinfo[3] = str(x0_out)
        mapinfo[4] = str(y0_out)
        metadata['map info'] = mapinfo
        metadata['lines'] = str(ymax - ymin)
        metadata['samples'] = str(xmax - xmin)
        if interleave is None:
            interleave = hypdata.metadata['interleave']
        else:
            hypdata.metadata['interleave'] = interleave
        localprintcommand(functionname + " creating file " + outfilename)
        outdata = envi.create_image(outfilename, metadata=metadata, interleave=interleave, ext='')
        outdata_map = outdata.open_memmap(writable=True)
        localprintcommand(" and saving data ... ")
        outdata_map[:, :, :] = hypdata_map[ymin:ymax, xmin:xmax, :]
        localprintcommand(" done .\n")  # actually, the file will be closed only after function exits


def figure2image(fig_hypdata, hypdata, hypdata_map, outfilename, interleave="bil", localprintcommand=None):
    """ save the zoomed area in fig_hypdata as a new ENVI file
    
    Args:
        fig_hypdata: matplotlib figure handle
        hypdata: the spectral python (SPy) file handle
        hypata_map: the data raster which is extracted and saved
        outfilename: the file to save to. 
        interleave: 'bil', 'bip, or 'bsq'
            if None, the interleave of input image is used 
        localprintcommand: function to use for output. If None, print() will be used
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)

    xlim = fig_hypdata.axes[0].get_xlim()
    ylim = fig_hypdata.axes[0].get_ylim()
    xmin = int(xlim[0] + 0.5)  # 0.5 to get rounding right
    xmax = int(
        xlim[1] + 1.5)  # 0.5 to get rounding right + 1 as the pixel at xmax will be the first to excluded from subset
    ymin = int(ylim[1] + 0.5)
    ymax = int(ylim[0] + 1.5)
    
    # imaxx = hypdata_map.shape[1] 
    # imaxy = hypdata_map.shape[0]

    localprintcommand("figure2image(): No pixels in the zoomed area. Aborting.\n")
    if xmin >= xmax or ymin >= ymax:
        localprintcommand("figure2image(): No pixels in the zoomed area. Aborting.\n")
    else:
        subset_raster(hypdata, outfilename, [xmin, ymin, xmax, ymax], hypdata_map, interleave,
                      localprintcommand=localprintcommand)
        localprintcommand("figure2image(): Saved " + outfilename + "\n")


def world2envi(envihdrfilename, pointmatrix):
    """
    convert the (usually projected) world coordiates in pointmatrix to the image 
    coordinates of envihdrfilename (relative to pixel center).
    pointmatrix: 2-column np.matrix [[x, y]]
    """
    GT = get_geotrans(envihdrfilename)

    # transform to hyperspectral figure coordinates
    X = pointmatrix[:, 0]
    Y = pointmatrix[:, 1]
    D = GT[1] * GT[5] - GT[2] * GT[4]
    xy = np.column_stack(((X * GT[5] - GT[0] * GT[5] + GT[2] * GT[3] - Y * GT[2]) / D - 0.5,
                          (Y * GT[1] - GT[1] * GT[3] + GT[0] * GT[4] - X * GT[4]) / D - 0.5))
    return xy


def envi2world(envihdrfilename, pointmatrix_local):
    """
    convert the image coordinates (relative to pixel center) in pointmatrix_local to the 
    (usually projected) world coordinates of envihdrfilename.
    pointmatrix_local: 2-column np.matrix [[x, y]]
    """
    GT = get_geotrans(envihdrfilename)

    # transform to hyperspectral figure coordinates
    P = pointmatrix_local[:, 0] + 0.5  # relative to pixel corner
    L = pointmatrix_local[:, 1] + 0.5
    xy = np.column_stack((GT[0] + P * GT[1] + L * GT[2],
                          GT[3] + P * GT[4] + L * GT[5]))

    return xy


def simulate_multispectral(hyp_infile, spectralsensitivityfile, out_pixelsize, out_multifilename=None, hypdata=None,
                           hypdata_map=None, pixelsizereserve=None, max_DIVfraction=0.1, out_interleave='bip',
                           multi_progressvar=None, localprintcommand=None):
    """ Joins pixels and bands of hyperspectral data to simulate a medium-resolution multispectral image.
    
    Implemented as a wrapper function around read_spectralsensitivity() + resample_hyperspectral()
    
    Args:
    hyp_infile: input hyperspectral data hdr file
    spectralsensitivityfile: file with spectral sensitivity functions. First column:wavlength, next columns sensitivity functions, 1 column per band
        spectral sensitivities will be renormalized to sum to unity over the hyperspectral bands
        Sentinel-2 spectral response functions can be downloaded from https://earth.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library/-/asset_publisher/Wk0TKajiISaR/content/sentinel-2a-spectral-responses
            xls: https://earth.esa.int/documents/247904/685211/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xlsx
    out_pixelsize: the pixel size of output data. 
        If out_pixelsize is a tuple, it is assumed to be one value per output band. The actual pixel size of the outputraster will be min(out_pixelsize)
        for pixels where s[i] = out_pixelsize[i]//max(out_pixelsize) > 1, n pixels will be averaged and have the same value (e.g., as for Sentinel 2)
    out_multifilename: the file name of the output multispectral data (will be in envi format)
            if None, output will not be saved to a file
    hypdata: the spectral python (SPy) file handle. If set, hyp_infile is not reopened.
            hypdata can aso be a metadata dictionary (=hypdata.metadata) if hypdata_map is a 3D np.ndarray-like object.  
    hypata_map: the data raster which is extracted. If hypdata is None, hyp_infile is opened and read.
    pixelsizereserve: the maximum pixel size geolocation has to account for. For example, Sentinel-2 has bands
        with different resolutions. All bands should be geolocated allowing for the largest, 60m pixel, achieved by pixelsizereserve=60
    max_DIVfraction: maximum number of allowed Data Ignore Values in a simulated pixel before the pixel itself is set to DIV
    out_interleave: interleave format of output data, can be bsq, bil or bip
    multi_progressvar: variable of type tkinter.DoubleVar() for tracking progress and disrupting processing
    localprintcommand: command for communicating to the user.
    
    Returns:
        3-dim np.ndarray-like object, either a memmap (if a file was opened) or a 3d array in memory
        metadata dictionary (similar to spectral envi ones)
    """

    if hypdata is None:
        hypdata_temp = spectral.open_image(hyp_infile)
        hyp_metadata = hypdata_temp.metadata
    elif type(hypdata) is dict:
        hyp_metadata = hypdata
    else:
        # hypdata has to be a Spectral Python file handle
        hyp_metadata = hypdata.metadata

    wavelength_stringlist = hyp_metadata['wavelength']  # a list of strings is returned
    # convert it to numpy array of floats
    hyp_wl = np.array(wavelength_stringlist, dtype=float)
    if hyp_wl.max() < 10:
        # likely, microns. convert to nanometers
        hyp_wl *= 1000

    # read the band sensitivity functions
    hyp_ss, multi_wl, b_subset = read_spectralsensitivity(spectralsensitivityfile, hyp_wl,
                                                          localprintcommand=localprintcommand)
    multi_bands = hyp_ss.shape[1]  # number of bands to simulate
    if not np.isscalar(out_pixelsize):
        out_pixelsize = out_pixelsize[b_subset]

    M, metadata = resample_hyperspectral(hyp_infile, hyp_ss, out_pixelsize, out_multifilename=out_multifilename,
                                         hypdata=hypdata, hypdata_map=hypdata_map,
                                         multi_bandcenters=multi_wl, pixelsizereserve=pixelsizereserve,
                                         max_DIVfraction=max_DIVfraction, out_interleave=out_interleave,
                                         multi_progressvar=multi_progressvar, localprintcommand=localprintcommand)

    return M, metadata


def read_spectralsensitivity(spectralsensitivityfile, hyp_wl=None, localprintcommand=None):
    """ read a spectral sensitivity file and calculates the weights for the (hyper)spectral channels which will be used to simulate this data 
    
    Args:
    spectralsensitivityfile: file with spectral sensitivity functions. First column:wavlength, next columns sensitivity functions, 1 column per band
        spectral sensitivities will be renormalized to sum to unity over the hyperspectral bands
        Sentinel-2 spectral response functions can be downloaded from https://earth.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library/-/asset_publisher/Wk0TKajiISaR/content/sentinel-2a-spectral-responses
            xls: https://earth.esa.int/documents/247904/685211/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xlsx
    hyp_wl: (hyper)spectral channels (in nm) which will be used to simulate this data
        if None, the original wavelengths in sensitivity file are used
    localprintcommand: command for communicating to the user.
    
    Returns: numpy arrays
        spectral sensitivity matrix (resampled to hyp_wl if hyp_wl set) (2d array)
        wavelength array (1d)
            * if wl_hyp set -- multispectral band central wavelengths (because this information may get lost in resampling)
            * if wl_hyp not set -- the wavelengths used in the spectral sensitivity file
        subset_index: which columns in the original file ended up in spectralsensitivityfile. 
    """

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)
    functionname = 'read_spectralsensitivity(): ' # for messaging
        
    # read the band sensitivity functions
    (M, headers) = readtextfile(spectralsensitivityfile)
    ss_wl = M[:, 0]
    if ss_wl.max() < 10:
        # likely, microns. convert to nanometers
        ss_wl *= 1000

    ss_M = M[:, 1:]
    multi_bands = ss_M.shape[1]  # number of multispectral bands

    if hyp_wl is None:
        subset_index = [int(i) for i in range(multi_bands)]
        return ss_M, ss_wl, subset_index
    else:
        if hyp_wl.max() < 10:
            # Likely, microns. Convert to nanometers
            hyp_wl_nm = hyp_wl * 1000
        else:
            hyp_wl_nm = hyp_wl
        hyp_wl_np = np.array(hyp_wl_nm)  # just in case -- now we know it's numpy.ndarray
        # interpolate the sensitivity values to hyperspectral bands
        hyp_ss = np.zeros((hyp_wl_np.shape[0], ss_M.shape[1]), dtype=float)

        # the wavelengths of the simulated bands, the location of maximum sensitivity
        multi_wl = np.zeros(multi_bands, dtype=float)
        for i in range(multi_bands):
            hyp_ss[:, i] = np.interp(hyp_wl_np, ss_wl, ss_M[:, i])
            # normalize to sum to 1
            if hyp_ss[:, i].sum() > 0:
                # there may be bands outside the range of hyperspectral data. 
                hyp_ss[:, i] /= hyp_ss[:, i].sum()
                multi_wl[i] = ss_wl[ss_M[:, i].argmax()]
        # which output bands can be created with the hyperspectral data?
        i_sen = np.where(multi_wl > 0)[0]
        hyp_ss = hyp_ss[:, i_sen]
        multi_wl = multi_wl[i_sen]
        return hyp_ss, multi_wl, i_sen


def resample_hyperspectral(hyp_infile, spectralsensitivitymatrix, out_pixelsize, out_multifilename=None,
                           hypdata=None, hypdata_map=None, multi_bandcenters=None, pixelsizereserve=None,
                           max_DIVfraction=0.1, out_interleave='bip',
                           spectralsensitivityfile=None, multi_progressvar=None, localprintcommand=None):
    """ joins pixels and bands of hyperspectral data to simulate a medium-resolution multispectral image
    
    Args:
    hyp_infile: input hyperspectral data hdr file
    spectralsensitivitymatrix: matrix with spectral sensitivity functions in columns, 1 column per band
        spectral sensitivities will NOT be renormalized 
    out_multifilename: the file name of the output multispectral data (will be in envi format)
        if None, output will not be saved to a file
    out_pixelsize: the pixel size of output data
        If out_pixelsize is a tuple, it is assumed to be one value per output band. The actual pixel size of the outputraster will be min(out_pixelsize)
        for pixels where s[i] = out_pixelsize[i]//max(out_pixelsize) > 1, n pixels will be averaged and have the same value (e.g., as for Sentinel 2)
    hypdata: the spectral python (SPy) file handle. If set, hyp_infile is not reopened.
        hypdata can aso be a metadata dictionary (=hypdata.metadata) if hypdata_map is a 3D np.ndarray-like object.  
    hypata_map: the data raster which is extracted. If hypdata is None, hyp_infile is opened and read.
    multi_bandcenters: the central wavelengths for each output band for saving in the outputfile header.
        if None, calculated from sensitivity matrix -- which is likely not very accurate.
    pixelsizereserve: the maximum pixel size geolocation has to account for. For example, Sentinel-2 has bands
        with different resolutions. All bands should be geolocated allowing for the largest, 60m pixel, achieved by pixelsizereserve=60
    max_DIVfraction: maximum number of allowed Data Ignore Values in a simulated pixel before the pixel itself is set to DIV
    out_interleave: interleave format of output data, can be bsq, bil or bip
    spectralsensitivityfile: the name of the file used for reading spectralsensitivitymatrix. For informational purposes only
    multi_progressvar: variable of type tkinter.DoubleVar() for tracking progress and disrupting processing
    localprintcommand: command for communicating to the user.
    
    Returns:
        3-dim np.ndarray-like object, either a memmap (if a file was opened) or a 3d array in memory
        metadata dictionary (similar to spectral envi ones)
    """

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)
    functionname = 'resample_hyperspectral(): ' # for messaging
    
    if hypdata is None:
        hypdata = spectral.open_image(hyp_infile)
        # open the file as a "memory map", a numpy matrix-like object. 
        hypdata_map = hypdata.open_memmap()

    if type(hypdata) is dict:
        # it is the metadata dictionary for hypdata_map
        hyp_metadata = hypdata
        # the file name is now just a decorative description and not used for any real purposes
    else:
        # hypdata has to be a spectral file handle
        #   read metadata from that file
        hyp_metadata = hypdata.metadata

    # see if we need to merge pixels for some bands (i.e., some bands have a larger pixel size)
    out_pixelsize_nominal = np.min(out_pixelsize)
    if np.isscalar(out_pixelsize):
        out_pixelsize_factor = np.ones_like(spectralsensitivitymatrix[0, :])
    else:
        out_pixelsize_factor = np.round(np.array(out_pixelsize) / out_pixelsize_nominal).astype(int)

    if pixelsizereserve == None:
        # make no space reservations
        pixelsizereserve = np.max(out_pixelsize_factor) * out_pixelsize_nominal
    elif pixelsizereserve < np.max(out_pixelsize_factor) * out_pixelsize_nominal:
        localprintcommand(functionname + " out_pixelsize_factor smaller than maximum pixelsize. Fixing\n")
        pixelsizereserve = np.max(out_pixelsize_factor) * out_pixelsize_nominal

    wavelength_stringlist = hyp_metadata['wavelength']  # a list of strings is returned
    # convert it to numpy array of floats
    hyp_wl = np.array(wavelength_stringlist, dtype=float)
    if hyp_wl.max() < 10:
        # likely, microns. convert to nanometers
        hyp_wl *= 1000

    multi_bands = spectralsensitivitymatrix.shape[1]  # number of bands to simulate
    if multi_bandcenters is None:
        multi_bandcenters = hyp_wl[spectralsensitivitymatrix.argmax(axis=0)]

    # create channel and weight lists for the simulated bands
    channellist = []
    weightlist = []

    for i in range(multi_bands):
        ii = np.where(spectralsensitivitymatrix[:, i] > 0)[0]
        channellist.append(ii)
        weightlist.append(spectralsensitivitymatrix[ii, i])

    # calculate the output file geometry. Make sure that the geographic coordinates of the upper-left corner of origin are divisible by 60
    # first, get the coordinates of pixel centers in hyperspectral data
    pixel_ind = np.array(range(hypdata.shape[1]))
    hyp_x = envi2world(hyp_infile, np.stack((pixel_ind, np.zeros_like(pixel_ind)), axis=1))[:, 0]
    pixel_ind = np.array(range(hypdata.shape[0]))
    hyp_y = envi2world(hyp_infile, np.stack((np.zeros_like(pixel_ind), pixel_ind), axis=1))[:, 1]
    hyp_pixelsize = get_pixelsize(hyp_infile)
    # find the upper-left corner of simulated multispectral data
    multi_x0 = np.ceil((hyp_x.min() - hyp_pixelsize[0] / 2) / pixelsizereserve) * pixelsizereserve
    multi_y0 = np.floor((hyp_y.max() + hyp_pixelsize[1] / 2) / pixelsizereserve) * pixelsizereserve
    # and the multispectral image dimensions
    multi_imageX = int((hyp_x.max() + hyp_pixelsize[0] / 2 - multi_x0) // out_pixelsize_nominal)
    multi_imageY = int((multi_y0 - hyp_y.min() - hyp_pixelsize[1] / 2) // out_pixelsize_nominal)

    # create lists of first and last hyperspectral channel indices in each simulated multispectral band
    pixelXlist_start = []
    pixelXlist_end = []
    for i in range(multi_imageX):
        # find hyperspectral pixels with centers inside multispectral pixels along x-axis
        x = multi_x0 + i * out_pixelsize_nominal
        ii = np.where(np.logical_and(hyp_x >= x, hyp_x < x + out_pixelsize_nominal))[0]
        pixelXlist_start.append(min(ii))
        pixelXlist_end.append(max(ii) + 1)  # for a range to include, it must be defined as range(i+1)

    # prepare metadata for the file to create
    multifile_metadata = {'bands': multi_bands, 'lines': multi_imageY, 'samples': multi_imageX,
                          'wavelength units': 'nm',
                          'interleave': out_interleave}

    multifile_metadata['description'] = "Simulated multispectral data from " + hyp_infile
    if spectralsensitivityfile is not None:
        multifile_metadata['description'] += ", bands from " + spectralsensitivityfile
    multifile_metadata['description'] += " of data: " + hyp_metadata['description']

    multifile_metadata['wavelength'] = ["%5.1f" % x for x in multi_bandcenters]

    use_DIV = False  # Use Data Ignore Value
    DIV = 0

    # DIV_testpixel = 0 # the band used to test NO DATA
    DIV_testpixel = int(
        hypdata_map.shape[2] / 2)  # the band used to test NO DATA: choose in the middle of the used hyp_wl range

    if 'data ignore value' in hyp_metadata:
        DIV = float(hyp_metadata['data ignore value'])
        multifile_metadata['data ignore value'] = DIV  # use always zero for decimal compatibility
        use_DIV = True

    # check for optional keys to include    
    for i_key in ['coordinate system string', 'data type' ]:
        if i_key in hyp_metadata:
            multifile_metadata[i_key] = hyp_metadata[i_key]

    multi_mapdata = hyp_metadata['map info']
    multi_mapdata[1:7] = [str(1), str(1), str(multi_x0), str(multi_y0), str(out_pixelsize_nominal),
                          str(out_pixelsize_nominal)]
    multifile_metadata['map info'] = multi_mapdata

    # create output matrix and file (if needed)
    if out_multifilename is None:
        multidata_map = np.zeros((multi_imageY, multi_imageX, multi_bands), dtype=type(hypdata_map[0, 0, 0]))
    else:
        multifile = envi.create_image(out_multifilename, metadata=multifile_metadata, ext='')
        multidata_map = multifile.open_memmap(writable=True)

    multi_pixels = multi_imageX * multi_imageY  # for progress report
    processed_pixels = 0
    progressincrement = 0.01  # the progress interval used in printing hash marks
    nextreport = 0  # the next progress value at which to print a hash mark
    break_signaled = False

    for i_sl, senline in enumerate(multidata_map):
        y_i = multi_y0 - i_sl * out_pixelsize_nominal
        ii = np.where(np.logical_and(hyp_y >= y_i - out_pixelsize_nominal, hyp_y < y_i))[0]
        i_Ystart = min(ii)
        i_Yend = max(ii) + 1  # for a range to include, it must be defined as range(i+1)
        for senpixel, j_Xstart, j_Xend in zip(senline, pixelXlist_start, pixelXlist_end):
            processed_pixels += 1
            if use_DIV:
                N = (j_Xend - j_Xstart) * (i_Yend - i_Ystart)
                frac_DIV = np.sum(hypdata_map[i_Ystart:i_Yend, j_Xstart:j_Xend, DIV_testpixel] == DIV) / N
                if frac_DIV > max_DIVfraction:
                    # too many Data Ignore Values, set the simulated pixel itself to DIV and go to next pixel
                    senpixel[:] = DIV
                    continue  # progress report will be skipped, but this will be a positive surprise to the user.
            for k, (channels, weights) in enumerate(zip(channellist, weightlist)):
                mean_hyp = hypdata_map[i_Ystart:i_Yend, j_Xstart:j_Xend, channels].mean(axis=(0, 1))
                senpixel[k] = sum(mean_hyp * weights)
            # report progress
            progresstatus = processed_pixels / multi_pixels
            if multi_progressvar is not None:
                if multi_progressvar.get() == -1:
                    # break signaled
                    break_signaled = True
                    break
                else:
                    # use the same variable to report progress
                    multi_progressvar.set(progresstatus)
            elif progresstatus > nextreport:
                localprintcommand("#")
                nextreport += progressincrement
        if break_signaled:
            break

    if np.max(out_pixelsize_factor) > 1:
        # some pixels are larger. Resample
        localprintcommand("\n" + functionname + " downscaling bands: ")
        for i_wl in np.where(out_pixelsize_factor > 1)[0]:
            f_i = out_pixelsize_factor[i_wl]
            localprintcommand("{:d}({:d}x) ".format(i_wl + 1, f_i))
            for j in range(0, multidata_map.shape[0], f_i):
                for i in range(0, multidata_map.shape[1], f_i):
                    multidata_map[j:j + f_i, i:i + f_i, i_wl] = np.mean(multidata_map[j:j + f_i, i:i + f_i, i_wl])

        localprintcommand("\n")
    if out_multifilename is not None:
        multidata_map.flush()

    return multidata_map, multifile_metadata


def sentinel2_pixelsize():
    """
    Return an ndarray with the pixel sizes of Sentinel-2 in meters
    """
    #                 B1  B2  B3  B4  B5  B6  B7  B8  B8A B9  B10 B11 B12
    return np.array( (60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20) )

def envihdr2datafile( hdrfilename, localprintcommand=None ):
    """
    try to locate the data file associated with the ENVI header file hdrfilename
    because gdal wants the name of the data file, not hdr
    
    Returns:
        the full filename of the data file
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)
    functionname = 'envihdr2datafile(): ' # for messaging
    
    # for envi files: gdal wants the name of the data file, not hdr
    hdrfilename_split = os.path.splitext( hdrfilename )
    
    if hdrfilename_split[1].lower() == ".hdr":
        datafilename = hdrfilename_split[0]
        if not os.path.exists(datafilename):
            # try different extensions, .dat and .bin and .bil and img
            basefilename = datafilename
            
            extensionlist = [ 'dat', 'bin', 'bil', 'bsq', 'bip', 'img' ]
            for extension in extensionlist:
                datafilename  = basefilename + '.'+ extension
                if os.path.exists(datafilename):
                    break
                else:
                    datafilename  = basefilename + '.'+ extension.upper()
                    if os.path.exists(datafilename):
                        break
    if not os.path.exists(datafilename):
        localprintcommand(functionname + "Cannot find the data file corresponding to {}.\n".format(hdrfilename) )
        datafilename = ''
    return datafilename
    

def envifilecomponents( filename_in, localprintcommand=None ):
    """
    Tries to guess the envi data file and header file names from filename_in
    filename_in is either data or header file
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)
    functionname = 'envifilecomponents(): ' # for messaging
    
    base_in, extension_in = os.path.splitext( filename_in)
    if  extension_in.lower() == ".hdr":
        headerfile = filename_in
        datafile = envihdr2datafile( headerfile, localprintcommand=localprintcommand  )
    else:
        # assume we were given the data file name
        datafile = filename_in
        headerfile = envidata2hdrfile( datafile, localprintcommand=localprintcommand )
    return datafile, headerfile

def envi_endiannesscode( aisa1_map ):
    """ Return the ENVI endianness-value (0 or 1) for numpy matrix.
    
    This information is a required field in ENVI header. Spectral python creates it automatically,
    Therefore, this function is largely redundant -- but keep it just in case.
    typical use would be metadata['byte order'] = envi_endiannesscode( aisa1_map )
    
    determine byte order to be saved in header:
    #   Byte order=0 is least significant  byte first (LSF) [==little-endian] 
    #      data (DEC and MS-DOS systems).
    #   Byte order=1 is most significant byte first (MSF) [==big-endian] data 
    #       (all other platforms).
    """
    if aisa1_map.dtype.byteorder == "<":
        # One of: ‘=’ native, ‘<’ little-endian, ‘>’ big-endian, ‘|’ not applicable
        endianness = 0
    elif aisa1_map.dtype.byteorder == ">":
        endianness = 1
    else:
        # it's either irrelevant, or, most likely, system-endian
        #  use system value -- according to ENVI documentation, a value is required
        if sys.byteorder == 'little':
            endianness = 0
        else:
            endianness = 1
    return endianness
    
def envi_addheaderfield( envifilename, fieldname, values, vectorfield=None, checkifexists=True, localprintcommand=None ):
    """ Adds a aline to ENVI header file
    
    ENVI file should be closed before rewriting.
    
    Args:
    envifilename: string, file name
    fieldname: name of the field to add
    values: the value to add. Can be a list, e.g. one per band
    vectorfield: (boolean) whether to save the field as a vector. None == automatic detection
    checkifexists: flag -- whether to stop if the field already exists
    """
    
    if vectorfield is None:
        if type(values) is list or type(values) is tuple:
            vectorfield = True
        else:
            vectorfield = False
    
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='',flush=True)
    functionname = 'envi_addheaderfield(): ' # for messaging

    datafile,hdrfile = envifilecomponents( envifilename, localprintcommand=localprintcommand )
    
    if checkifexists and fieldname in open(hdrfile).read() :
        localprintcommand( functionname +" field <{}> already exists in {}. Stopping.\n"
            .format( fieldname, hdrfile ) )
    else:
        with open(hdrfile,'a') as hfile:
            if vectorfield:
                valuestr = [ str(i) for i in values ]
                outstr = fieldname + " = {" + ", ".join(valuestr) + "}"
            else:
                outstr = fieldname + " = " + str(values)
            hfile.write( outstr + "\n" )
        fieldtype = 'as vector' if vectorfield else ''
        localprintcommand( functionname +" Added field <{}> {} to {}.\n"
            .format( fieldname, fieldtype, hdrfile ) )





