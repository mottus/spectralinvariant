# GUI to pick pixel values in either one image or all images in a folder
import numpy as np
import spectral
import spectral.io.envi as envi
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import *
import copy
import os
import math
import matplotlib.pyplot as plt
import matplotlib.path 
from osgeo import ogr
from osgeo import osr
import matplotlib
import sys

from GUI_pointsfromband import PfBGUI
from spectralinvariant.hypdatatools_gdal import *
from spectralinvariant.hypdatatools_utils import *
from spectralinvariant.hypdatatools_img import *

class pixelGUI:
        
    def __init__( self, master ):
        
        self.master = master
        self.polygonlist = [] #  A list of GDAL geometries containing the polygons (rings in POLYGONs)
        self.polygonIDlist = [] #  A list names (IDs) for each polygon in the poygonlist
        #  Currently, use of just 1 polygon is implemented

        self.pointlist = [] # list of points, each point is a tuple: ( id, x, y ) (in global projected coordinates )

        self.openfilelist = [] # list of loaded file names and handles, reflects the contents of self.listbox_files
            # each element is a list [ filename filehandle datahandle DataIgnoreValue ]
            # intially, when loading the hyperspectral handles are set to None; they are assigned when file is opened for e.g. plotting

        # the following list contains information on open windows. updated whenever possible
        self.figurelist = [] # each list element should be a list: [ number, type, figurehandle, name ,full_filename , DataIgnoreValue ]
        # type: 'hyp' -- hyperspectral (spectral) [no other defined or used]
        
        # the colors to loop through. Not a good idea, self.plotcolors will likely not be used anymore soon
        # self.plotcolors =  ( 'r', 'b', 'g', 'c', 'm', 'y', 'k', 'w' ) 
        self.plotcolors =  ( 'r' ) 
        self.plotlinestyles = ( '-', ':' , '-.', '--' )
        self.plotlinecolors =  ( 'r', 'b', 'g', 'c', 'm', 'y', 'k', 'w' )
        
        # the brightness stretch scale factor, colors above stretchfactor are set to 1
        self.stretchfactor_string = StringVar()
        self.stretchfactor_string.set( "0.95" )
        
        self.catch_figure = None # The figure handle from which clicks are being caught
        self.catch_figure_listno = -1 # the number of figure in figurelist where clicks are captured
        self.hypdata_ciglock = False # cig lock for fig_hypdata so only one function can catch clicks from pyplot
        self.catch_cid = -1

        bw = 25 # buttonwidth
        ow = bw-10 # optionmenuwidth. bw-5 produces a width alost identical to button
        
        # Show the GUI in a Toplevel window instead of Root. 
        # This allows many such programs to be run independently in parallel.
        self.w = Toplevel( master )
        self.w.title("GUI for selecting pixels")
        
        self.areashape_string = StringVar() # string to set and read option_areashape OptionMenu
        self.areaunit_string = StringVar() # string to set and read option_areaunit OptionMenu
        self.plotmode_string = StringVar() # string to set and read option_plotmode OptionMenu
        areashape_list = [ 'Select square, side' , 'Select circle, d=' ]
        areaunit_list = [ 'pixels' , 'meters' ]
        plotmode_list = [ 'default' , 'RGB' , 'NIR' , 'falsecolor' ]

        self.areasize_string = StringVar() # the string to be read as input from user via Entry entry_areasize
        self.areashape_string.set( areashape_list[0] )
        self.areaunit_string.set( areaunit_list[0] )
        self.areasize_string.set("3")
        self.plotmode_string.set( plotmode_list[0] )
        
        pythonplotcolors_list =  [ 'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white' ]
        self.pointcolor_string = StringVar() # string for the OptionMenu to set mark color
        self.pointcolor_string.set( pythonplotcolors_list[0] )
        
        self.textlog = ScrolledText( self.w, height=6 )
        self.textlog.pack( side='bottom' )
        
        self.frame_button = Frame( self.w )
        self.button_quit = Button( self.frame_button, width=bw, text='Quit', command=self.buttonquit_fun )
        self.button_loadpoints = Button( self.frame_button, width=bw, text='Load points from .txt', command=self.loadpoints_fun )
        self.button_loadfile = Button( self.frame_button, width=bw, text='Load raster files', command=self.loadfiles_fun )
        self.button_pixelvalue = Button( self.frame_button, width=bw, text='Store spectrum values', command=self.pixelvalue_fun, state=DISABLED )
        self.button_plotspectra = Button( self.frame_button, width=bw, text='Plot spectra', command=self.plotspectra_fun, state=DISABLED )
        self.button_analyzepoints = Button( self.frame_button, width=bw, text='Analyze spectra', command=self.analyzepoints_fun, state=DISABLED )
        self.button_loadshp = Button( self.frame_button, width=bw, text='Load polygons from .shp', command=self.loadshp_fun )
        self.button_plotshp = Button( self.frame_button, width=bw, text='Draw polygons', command=self.plotshp_fun, state=DISABLED )
        self.button_zoomtoshp = Button( self.frame_button, width=bw, text='Zoom to polygon', command=self.zoomtoshp_fun, state=DISABLED )
        self.button_pointsfromband = Button( self.frame_button, width=bw, text='Points from band threshold', command=self.pointsfromband_fun, state=DISABLED )
        self.option_areashape = OptionMenu( self.frame_button, self.areashape_string, *areashape_list )
        self.option_areashape['width'] = ow
        self.option_areaunit = OptionMenu( self.frame_button, self.areaunit_string, *areaunit_list )
        self.option_areaunit['width'] = ow
        self.entry_areasize = Entry( self.frame_button, textvariable=self.areasize_string )
        self.button_loadpoints.pack( side='top' )
        self.button_loadfile.pack( side='top' )
        self.option_areashape.pack( side='top' )
        self.entry_areasize.pack( side='top' )
        self.option_areaunit.pack( side='top' )
        self.button_pixelvalue.pack( side='top' )
        self.button_plotspectra.pack( side = 'top' )
        self.button_analyzepoints.pack( side = 'top' )
        self.button_loadshp.pack( side='top' )
        self.button_plotshp.pack( side='top' )
        self.button_zoomtoshp.pack( side='top' )
        self.button_pointsfromband.pack( side='top' )
        self.button_quit.pack( side='bottom' )
        self.frame_button.pack( side='left' )
        
        self.frame_files = Frame( self.w )
        self.scrollbar_files_v = Scrollbar( self.frame_files, orient='vertical' )
        self.scrollbar_files_h = Scrollbar( self.frame_files, orient='horizontal' )
        self.listbox_files = Listbox( self.frame_files, exportselection=False, selectmode='extended', yscrollcommand=self.scrollbar_files_v.set, xscrollcommand=self.scrollbar_files_h.set )
        # or selectmode "multiple"?
        self.scrollbar_files_v['command'] = self.listbox_files.yview
        self.scrollbar_files_h['command'] = self.listbox_files.xview
        self.label_files = Label( self.frame_files, width=bw, text="RASTER FILES")
        self.button_deletefile = Button( self.frame_files, width=bw, text='Remove from list', command=self.deletefiles_fun, state=DISABLED )
        self.option_plotmode = OptionMenu( self.frame_files, self.plotmode_string, *plotmode_list )
        self.option_plotmode['width'] = ow
        self.label_stretchfactor = Label( self.frame_files, width=bw, text="stretch factor")
        self.entry_stretchfactor = Entry( self.frame_files, textvariable=self.stretchfactor_string )
        self.button_displayfile = Button( self.frame_files, width=bw, text='Display', command=self.displayfile_fun, state=DISABLED )
        
        self.button_displayfile.pack( side='bottom' )
        self.entry_stretchfactor.pack( side='bottom' )
        self.label_stretchfactor.pack( side='bottom' )
        self.option_plotmode.pack( side='bottom' )
        self.button_deletefile.pack( side='bottom' )
        self.label_files.pack( side='top' )
        self.scrollbar_files_v.pack( side='right', fill='y' )
        self.scrollbar_files_h.pack( side='bottom', fill='x' )
        self.listbox_files.pack( side='top', fill=BOTH, expand=True )
        self.frame_files.pack( side='left', fill=BOTH, expand=True  )

        self.frame_points = Frame( self.w )
        self.scrollbar_points_h = Scrollbar( self.frame_points, orient='horizontal' )
        self.scrollbar_points_v = Scrollbar( self.frame_points, orient='vertical' )
        self.label_points = Label( self.frame_points, width=bw, text="POINTS")
        self.listbox_points = Listbox( self.frame_points, exportselection=False, selectmode='extended', xscrollcommand=self.scrollbar_points_h.set, yscrollcommand=self.scrollbar_points_v.set )
        self.scrollbar_points_h['command'] = self.listbox_points.xview
        self.scrollbar_points_v['command'] = self.listbox_points.yview
        self.button_zoomtopoints = Button( self.frame_points, width=bw, text='Zoom to', command=self.zoomtopoints_fun, state=DISABLED )
        self.option_pointcolor = OptionMenu( self.frame_points, self.pointcolor_string, *pythonplotcolors_list )
        self.option_pointcolor['width'] = ow
        self.button_showpoints = Button( self.frame_points, width=bw, text='Show in raster', command=self.showpoints_fun, state=DISABLED )
        self.button_addpoint = Button( self.frame_points, width=bw, text='Add with mouse', command=self.addpoint_fun, state=DISABLED )
        self.button_deletepoints = Button( self.frame_points, width=bw, text='Delete', command=self.deletepoints_fun, state=DISABLED )
        self.button_savepoints = Button( self.frame_points, width=bw, text='Save selected to txt file', command=self.savepoints_fun, state=DISABLED )
        self.button_updatepoints = Button( self.frame_points, width=bw, text='Update list', command=self.updatepoints_fun, state=DISABLED )
        self.button_selectzoomedpoints = Button( self.frame_points, width=bw, text="Select points in zoom", command=self.selectpoints_zoom_fun, state=DISABLED )
        
        self.button_zoomtopoints.pack( side='bottom' )
        self.button_selectzoomedpoints.pack( side='bottom' )
        self.option_pointcolor.pack( side='bottom' )
        self.button_showpoints.pack( side='bottom' )
        self.button_savepoints.pack( side='bottom' )
        self.button_deletepoints.pack( side='bottom' )
        self.button_addpoint.pack( side='bottom' )
        self.button_updatepoints.pack( side='bottom' )
        self.label_points.pack( side='top' )
        self.scrollbar_points_v.pack( side='right', fill='y' )
        self.scrollbar_points_h.pack( side='bottom', fill='x' )
        self.listbox_points.pack( side='top', fill='y', expand=True )
        self.frame_points.pack( side='right', fill='y')
        
        self.frame_figures = Frame( self.w )
        self.scrollbar_figures_h = Scrollbar( self.frame_figures, orient='horizontal' )
        self.scrollbar_figures_v = Scrollbar( self.frame_figures, orient='vertical' )
        # self.listbox_figures = Listbox( self.frame_figures, exportselection=False, selectmode='single', xscrollcommand=self.scrollbar_figures_h.set, yscrollcommand=self.scrollbar_figures_v )
        self.listbox_figures = Listbox( self.frame_figures, exportselection=False, selectmode='extended', xscrollcommand=self.scrollbar_figures_h.set, yscrollcommand=self.scrollbar_figures_v )
        self.scrollbar_figures_h['command'] = self.listbox_figures.xview
        self.scrollbar_figures_v['command'] = self.listbox_figures.yview
        self.label_figures = Label( self.frame_figures, width=bw, text="FIGURES")
        self.button_updatefigures = Button( self.frame_figures, width=bw, text='Update list', command=self.update_figures_fun, state=ACTIVE )
        self.button_clearfigure = Button( self.frame_figures, width=bw, text='Clear markings', command=self.clearfigure_fun, state=DISABLED )
        self.button_zoomout = Button( self.frame_figures, width=bw, text='Zoom out 2x', command=self.zoomout_fun, state=DISABLED )
        self.button_zoomfull = Button( self.frame_figures, width=bw, text='Zoom full', command=self.zoomfull_fun, state=DISABLED )
        self.button_zoomtofile = Button( self.frame_figures, width=bw, text='Zoom to selected file', command=self.zoomtofile_fun, state=DISABLED )
        self.button_savezoomarea = Button( self.frame_figures, width=bw, text='Save zoomed area to file', command=self.savezoomed_fun, state=DISABLED )
        self.button_closefigure = Button( self.frame_figures, width=bw, text='Close', command=self.closefigure_fun, state=DISABLED )
        self.button_closefigure.pack( side='bottom' )
        self.button_savezoomarea.pack( side='bottom' )
        self.button_zoomtofile.pack( side='bottom' )
        self.button_zoomout.pack( side='bottom' )
        self.button_zoomfull.pack( side='bottom' )
        self.button_clearfigure.pack( side='bottom' )
        self.button_updatefigures.pack( side='bottom' )
        self.label_figures.pack( side='top' )
        self.scrollbar_figures_v.pack( side='right', fill='y' )
        self.scrollbar_figures_h.pack( side='bottom', fill='x' )
        self.listbox_figures.pack( side='top', fill='y', expand=True )
        self.frame_figures.pack( side='right', fill='y' )
        self.foldername1 = get_hyperspectral_datafolder( localprintcommand=self.printlog ) # where the data is. This is the initial value, will be modified later
        
        # catch the signal that the point list has been updated by PfBGUI
        master.bind("<<PfBGUI_exit>>", self.updatepoints_fun )
        
    def printlog( self , text ):
        """
        Output to log window. Note: no newline added beteen inputs.
        text need not be a string, will be converted when printing.
        """
        self.textlog.insert( END, str(text) )
        self.textlog.yview( END )
        self.master.update_idletasks()        
        
    def loadpoints_fun( self ):
        """
        function to load the file of data points into the listbox
        assumes formt [x,y,id]
        but checks the first row for possible column titles (with limited intelligence)
        """
        # load csv, beware of possible uncommented column headings
        filename =  filedialog.askopenfilename(initialdir = self.foldername1, title = "Choose a file with pixel coordinates (X,Y)",filetypes = (("txt files","*.txt"),("csv files","*.csv"),("all files","*.*")))
        if filename!="":
            readxy=False 
            # try different separators numbers of header rows 
            # read as unicode text, so maybe it's all unnecessary?
            for rowstoskip in range(3):
                for sep in (';',',','\t',' '):
                    try:
                        xy = np.loadtxt( filename, skiprows=rowstoskip, delimiter=sep, dtype='unicode' )
                    except ValueError:
                        pass
                    else:
                        if len( xy.shape ) > 1 :
                            # stop only if more than one column is retrieved in xy
                            self.printlog( filename + ": using separator [" + sep + "].\n")
                            readxy=True
                            break
                if readxy:
                    break

            # test for column names in first row
            column_names = xy[0,:]
            
            if column_names[0][0] == "#":
                column_names[0] = column_names[0][1:] # header line might be commented out
            
            i_x = -1 # the column number for x coordinates
            i_y = -1 # the column number for y coordinates
            i_id = -1 # the column number for x coordinates
            
            # try to locate the proper columns
            for i,c in enumerate(column_names):
                if len(c) > 0:
                    firstchar = c[0].lower() 
                    if firstchar == 'x' and i_x==-1:
                        i_x = i
                        self.printlog( filename + ": Found column for X: %i %s.\n" %(i_x+1,c) )
                    elif firstchar == 'y' and i_y==-1:
                        i_y = i
                        self.printlog( filename + ": Found column for Y: %i %s.\n" %(i_y+1,c) )
                    elif firstchar == 'i' and i_id==-1:
                        i_id = i
                        self.printlog( filename + ": Found column for ID: %i %s.\n" %(i_id+1,c) )
            
            if (i_x == -1) or ( i_y == -1 ):
                self.printlog( filename + ": Could not identify X,Y columns. Setting 1st=X ,2nd=Y.\n")
                # use default values
                i_x = 0 # xcoordinates of points
                i_y = 1 # y coordinates of points
                
            if ( i_id == -1 ):
                if xy.shape[1] > 2:
                    i_id = 2
                    self.printlog( filename + ": Using 3rd column as ID.\n")
                else:
                    i_id = None
                    self.printlog( filename + ": Using counter as ID.\n")
            
            # read x,y. We need to determine the rows with numerical coordinates
            #   empty rows should be discarded
            x_in = []
            y_in = []
            id_in = []
            counter_rows = 0
            for ii in range( xy.shape[0] ):
                try:
                    # check only x-coordinate, assume y is OK if x is.
                    x_temp = np.float32( xy[ ii, i_x ] )
                except ValueError:
                    pass
                else:
                    counter_rows += 1
                    x_in.append( x_temp )
                    # assume y coordinate is OK, too
                    y_in.append( np.float32( xy[ ii, i_y ] ) )
                    if i_id is not None:
                        id_in.append( str( xy[ ii, i_id ] ) )
                    else:
                        # use just sequential numbers as IDs
                        id_in.append( str( counter_rows ) )
    
            # store the points in listbox and self.pointlist
            # select format based on their values
            if x_in[0] < 500:
                #likely, in degrees, give high accuracy
                xfmt = '{:.5f}'
            else:
                # likely, in meters, 10 cm is already an overkill
                xfmt = '{:.1f}'
            # just in case, although it's likely very safe to use the same format for x and y
            if y_in[0] < 500:
                yfmt = '{:.5f}'
            else:
                yfmt = '{:.1f}'
            self.xy_arr = []
            xy_str = []

            #clear the listbox of any existing data
            # self.listbox_points.delete( 0, END )            
            # self.pointlist=[]
            
            olditems = len( self.pointlist )
            for x,y,id in zip(x_in,y_in,id_in):
                xy_str.append( id+","+xfmt.format(x)+","+yfmt.format(y) )
                self.xy_arr.append( [str(x),str(y),id] )
                self.pointlist.append( ( id, float(x), float(y) ) )
            # populate the listbox
            for item in xy_str:
                self.listbox_points.insert( END, item )
                
            # select newly inserted points
            self.listbox_points.selection_clear( 0, END )
            self.listbox_points.selection_set( olditems, END )
            self.listbox_points.see( olditems )
            
            self.pointsloaded = True
            self.button_loadpoints.configure( background='green' )
            self.button_deletepoints.configure( state=ACTIVE )
            self.button_savepoints.configure( state=ACTIVE )
            self.button_updatepoints.configure( state=ACTIVE )
            if len(self.openfilelist)>0:
                self.button_pixelvalue.configure( state=ACTIVE )
                self.button_plotspectra.configure( state=ACTIVE )
                self.button_analyzepoints.configure( state=ACTIVE )
                if len( self.figurelist ) > 0:
                    self.button_zoomtopoints.configure( state=ACTIVE )
                    self.button_selectzoomedpoints.configure( state=ACTIVE )
                    self.button_showpoints.configure( state=ACTIVE )
                    self.button_addpoint.configure( state=ACTIVE )
        else:
            self.printlog("loadpoints_fun(): loading of points aborted.\n")
            
    def savezoomed_fun( self ):
        """ 
        save the area corresponding to current zoom as a new ENVI file
        """
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) == 0:
            self.printlog("savezoomed(): No figure selected, nothing to save.\n")
        else:
            curfigure = curfigurelist[0]
            fig_hypdata = self.figurelist[ curfigure ][ 2 ]
            hypfilename = self.figurelist[ curfigure ][ 4 ]
            hypdata = None
            # see if this file is open. If not, open.
            for i, fh in enumerate(self.openfilelist):
                if fh[0] == hypfilename:
                    # make sure it is open
                    self.openhypfile( i )
                    hypdata = fh[1]
                    hypdata_map = fh[2]
            
            if hypdata is None:
                # the file was not on the list. strange, but maybe possible
                hypdata = spectral.open_image( hypfilename )
                hypdata_map = hypdata.open_memmap()
                
            outfilename =  filedialog.asksaveasfilename(initialdir = self.foldername1, title = "Save zoomed area to...", filetypes = (("ENVI hdr files","*.hdr"),("all files","*.*")))
            
            self.printlog("savezoomed(): Saving data in the first selected figure, " + str(curfigure) + " to " + outfilename + ".\n")
            figure2image( fig_hypdata, hypdata, hypdata_map, outfilename, localprintcommand=self.printlog )
            
    def zoomout_fun( self ):
        """
        zoom out by a factor of 2x2
        """
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) > 0:
            for curfigure in curfigurelist:
                fig_hypdata_number = self.figurelist[ curfigure ][ 0 ] 
                if fig_hypdata_number in plt.get_fignums():
                    # zoom out
                    fig_hypdata = self.figurelist[ curfigure ][ 2 ] 
                    xlim = fig_hypdata.axes[0].get_xlim()
                    ylim = fig_hypdata.axes[0].get_ylim()
                    xrange = xlim[1] - xlim[0]
                    yrange = ylim[1] - ylim[0]
                    minx_i = xlim[0] - xrange/2
                    maxx_i = xlim[1] + xrange/2
                    miny_i = ylim[0] - yrange/2
                    maxy_i = ylim[1] + yrange/2
                    
                    fig_hypdata.axes[0].set_xlim( ( minx_i, maxx_i ) )
                    fig_hypdata.axes[0].set_ylim( ( miny_i, maxy_i ) )
                    fig_hypdata.canvas.draw()
        # just in case 
        self.update_figures_fun()

    def zoomfull_fun( self ):
        """
        zoom to full extent of image
        """
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) > 0:
            for curfigure in curfigurelist:
                fig_hypdata_number = self.figurelist[ curfigure ][ 0 ] 
                if fig_hypdata_number in plt.get_fignums():
                    # zoom full: find file dimensions
                    fig_hypdata = self.figurelist[ curfigure ][ 2 ]
                    full_filename = self.figurelist[ curfigure ][ 4 ]
                    hypdata_map = None
                    # find file name in openfilelist
                    # If not found, autoscale will be used.
                    for fe in self.openfilelist:
                        if len( fe ) > 0:
                            if fe[0] == full_filename:
                                hypdata_map = fe[2]
                                break
                    zoomtoimage( fig_hypdata, hypdata_map )
        else:
            self.printlog("zoomfull(): no image selected, nothing to do.\n")

        # just in case 
        self.update_figures_fun()
        
    def zoomtofile_fun( self ):
        """
        zoom to the full extent of an image in another file
        """
        functionname = "zoomtofile():"
                
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) > 0:
            messageprinted = False # for keeping track of messages
            for curfigure in curfigurelist:
                fig_hypdata_number = self.figurelist[ curfigure ][ 0 ] 
                if fig_hypdata_number in plt.get_fignums():
                    # Figure still around, zoom it!
                    fig_hypdata = self.figurelist[ curfigure ][ 2 ]
                    hypfilename = self.figurelist[ curfigure ][ 4 ]
                    figurenumber = self.figurelist[ curfigure ][ 0 ]
                    
                    # get the first selected file name
                    if len( self.listbox_files.curselection() ) == 0:
                        # I am not sure if this is even possible -- no selection
                        self.printlog(functionname+" no file selected, nothing to zoom to.\n")
                    else:
                        # get the first file
                        selectedfile = self.listbox_files.curselection()[0]
                        # open the file if not open yet
                        filename_short2 = self.listbox_files.get(selectedfile) # file name without path or extension
                        hypfilename2 = self.openfilelist[selectedfile][0]
                        if not messageprinted:
                            self.printlog( functionname + "zooming figure ")
                            messageprinted = True
                        self.printlog( "%i " % figurenumber )                        
                        # find file dimensions and create a vector of corner points
                        imagesize = get_imagesize( hypfilename2 )
                        # pixel centers of corner pixels in image coordinates of the external image
                        corner_arr2 = np.array( [ [0,0] , [ imagesize[0]-1,imagesize[1]-1 ] ] )
                        # pixel centers of corner pixels in global coordinates
                        corner_arrglob = image2world( hypfilename2, corner_arr2 )
                        # pixel centers of corner pixels in local coordinates of the image in the figure
                        corner_arr = world2image( hypfilename, corner_arrglob )
                        
                        minx = corner_arr[0,0]
                        maxx = corner_arr[1,0]
                        miny = corner_arr[0,1]
                        maxy = corner_arr[1,1]
                        fig_hypdata.axes[0].set_xlim( ( minx-0.5, maxx-0.5 ) )
                        fig_hypdata.axes[0].set_ylim( ( maxy-0.5, miny-0.5 ) )
                        fig_hypdata.canvas.draw()
            if messageprinted:
                self.printlog( "to extent of file %s.\n" % filename_short2  )
            else:
                self.printlog( functionname +" no figures found.\n" )
        else:
            self.printlog(functionname + " no image selected, nothing to do.\n")

        # just in case 
        self.update_figures_fun()
                    
    def updatepoints_fun( self, *args ):
        """
        update the listbox containing points. Not done automatically because of a mysterious python crash
        """
        self.listbox_points.delete(0,END)
        if len( self.pointlist ) > 0:
            for point in self.pointlist:
                xp_str = "%1.1f" % point[1] # convert to string with one decimal
                yp_str = "%1.1f" % point[2]
                self.listbox_points.insert( END, point[0]+ ',' + xp_str + ',' + yp_str )
            self.button_zoomtopoints.configure( state=ACTIVE )
            self.button_selectzoomedpoints.configure( state=ACTIVE )
            self.button_showpoints.configure( state=ACTIVE )
            self.button_deletepoints.configure( state=ACTIVE )
            self.button_savepoints.configure( state=ACTIVE )
            self.button_plotspectra.configure( state=ACTIVE )
            self.button_analyzepoints.configure( state=ACTIVE )
            self.button_pixelvalue.configure( state=ACTIVE )
            self.update_figures_fun() # this should be called as often as possible
            
    def addpoint_fun( self, event=None ):
        """
        catch clicks in figure self.fig_hypdata and adds a new point    
        action depends on the state of the button self.button_addpoint
        """
        if self.button_addpoint.cget('text') == 'Add with mouse':
            # initiate pixel location collection, set up connection with figure window
            if not self.hypdata_ciglock: # only do sth if no other function is waiting for a click in fig_hypdata
                curfigurelist = self.listbox_figures.curselection() # get the current figure
                if len( curfigurelist ) > 0:
                    curfigure = curfigurelist[0]
                    fig_hypdata_number = self.figurelist[ curfigure ][ 0 ] 
                    if fig_hypdata_number in plt.get_fignums():
                        # now we should be all set: there is a selected figure which actually exists
                        # set up GUI
                        self.button_deletepoints.configure( state=DISABLED )
                        self.button_showpoints.configure( state=DISABLED )
                        self.button_zoomtopoints.configure( state=DISABLED )
                        self.button_selectzoomedpoints.configure( state=DISABLED )
                        self.button_savepoints.configure( state=DISABLED )
                        self.button_updatepoints.configure( state=DISABLED )
                        self.listbox_points.delete( 0, END )
                        self.listbox_points.insert( END, 'Click "Update list"' )
                        # get ready to catch click
                        fig_hypdata = self.figurelist[ curfigure ][ 2 ] 
                        self.button_addpoint.configure( text='CLICK IN FIGURE '+str(fig_hypdata_number), background='red' )
                        self.hypdata_ciglock = True
                        self.catch_figure = fig_hypdata
                        self.catch_figure_listno = curfigure
                        self.catch_cid = self.catch_figure.canvas.mpl_connect('button_press_event',self.addpoint_fun)
                    else:
                        self.printlog("addpoint_fun(): Figure does not seem to exist anymore. Aborting adding point.\n")
                else:
                    self.printlog("addpoint_fun(): No figure selected. This should not be possible. Aborting adding point.\n")
            else:
                self.printlog("addpoint_fun(): Already catching clicks from figure " + str(self.catch_figure.number) +".\n" )
        elif event==None:
            # button text was 'CLICK IN IMAGE', but the button itself was clicked.
            # probably, the user wants to cancel
            self.printlog("addpoint_fun(): canceling pixel selection.\n")
            try:
                # tkinter may have a bug and SystemButtonFace would give an error, use gray85 instead
                self.button_addpoint.configure( text='Add with mouse', background='SystemButtonFace') 
            except TclError:
                self.button_addpoint.configure( text='Add', background='gray85')
            self.button_updatepoints.configure( state=ACTIVE )
            self.catch_figure.canvas.mpl_disconnect(self.catch_cid)
            # self.catch_figure = None # release lock on cig for fig_hypdata  XXX CAN CRASH PYTHON!
            self.hypdata_ciglock = False
            self.catch_figure_listno = -1
        else: 
            # we are called by pyplot event, a click in fig_hypdata
            self.printlog("addpoint_fun(): clicked "+str(event.xdata)+','+str(event.ydata)+".\n")

            hypfilename = self.figurelist[ self.catch_figure_listno ][ 4 ] 
            fig_hypdata = self.figurelist[ self.catch_figure_listno ][ 2 ]
                       
            # convert x,y to geographic coordinates           
            xy = image2world( hypfilename, np.asmatrix( (event.xdata,event.ydata) ) )
            
            # create id number: first unused number which is larger than the current number of loaded points
            # first, go through all points in pointlist
            existingids = [ x[0] for x in self.pointlist ]
            newid = str( len( self.pointlist )+1 )
            while newid in existingids:
                newid = str( int(newid)+1 )            
            
            self.printlog( "addpoint_fun() adding point %s: %0.1f,%0.1f (%0.1f,%0.1f)\n" 
                % (newid, xy[0,0], xy[0,1], event.xdata, event.ydata) )
            self.pointlist.append( (newid, xy[0,0], xy[0,1] ) )

            # mark in figure
            col = int( newid ) % len( self.plotcolors )
            fig_hypdata.axes[0].plot( event.xdata, event.ydata, marker='x', c=self.plotcolors[col] )
            fig_hypdata.axes[0].annotate( newid, (event.xdata+1, event.ydata-1 ), color=self.plotcolors[col] )
            try:
                # tkinter may have a bug and SystemButtonFace would give an error, use gray85 instead
                self.button_addpoint.configure( text='Add with mouse', background='SystemButtonFace') 
            except TclError:
                self.button_addpoint.configure( text='Add with mouse', background='gray85')

            self.button_updatepoints.configure( state=ACTIVE )
            
            self.catch_figure.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1
            # self.catch_figure = None # release lock on cig for fig_hypdata  XXX CAN CRASH PYTHON!
            self.hypdata_ciglock = False
            catch_figure_listno = -1
            
        # just in case 
        # self.update_figures_fun() # WILL CRASH PYTHON
    
    def deletepoints_fun( self ):
        """
        delete the selected points
        """
        selectedpoints = self.listbox_points.curselection()
        if len( selectedpoints ) > 0:
            for i in reversed(selectedpoints):
                self.pointlist.pop(i)
            self.updatepoints_fun()
        else:
            self.printlog("deletepoints_fun(): Nothing selected, nothing deleted.\n")
        self.update_figures_fun() # this should be called as often as possible

    def selectpoints_zoom_fun( self ):
        """
        select points which are in the zoomed part of the window
        """
        functionname = "selectpoints_zoom()"
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) == 0:
            self.printlog( functionname+": No figure selected, nothing to save.\n" )
        else:
            curfigure = curfigurelist[0]
            fig_hypdata = self.figurelist[ curfigure ][ 2 ]
            hypfilename = self.figurelist[ curfigure ][ 4 ]
            xlim = fig_hypdata.axes[0].get_xlim()
            ylim = fig_hypdata.axes[0].get_ylim()
            xmin = int( xlim[0]+0.5 )
            xmax = int( xlim[1]+1.5 )
            ymin = int( ylim[1]+0.5 )
            ymax = int( ylim[0]+1.5 )
    
            # clear selection
            self.listbox_points.selection_clear( 0, END )
            # convert pointlist to two-column nd.array and then to image coordinates
            pointlistarr = self.pointlist2matrix()
            xymatrix = world2image( hypfilename, pointlistarr )

            for i,xyrow in enumerate(xymatrix):
                if xyrow[0,0]>xmin and xyrow[0,0]<xmax and xyrow[0,1]>ymin and xyrow[0,1]<ymax:
                    self.listbox_points.selection_set( i )
        self.update_figures_fun() # this should be called as often as possible


    def savepoints_fun( self ):
        """
        save the selected points (i.e., their coordinates, not spectra) to a text file.
        """
        selectedpoints = self.listbox_points.curselection()
        if len( selectedpoints ) > 0:
            filename =  filedialog.asksaveasfilename(initialdir = self.foldername1, title = "Save point coordinates (X,Y)",filetypes = (("txt files","*.txt"),("csv files","*.csv"),("all files","*.*")))
            if filename != '':
                with open(filename,'w') as file:
                    file.write("id,x,y\n")
                    for i in selectedpoints:
                        point = self.pointlist[i]
                        pointstring = point[0] + "," + str(point[1]) + ',' + str(point[2]) + '\n'
                        file.write( pointstring )
            else:
                self.printlog("Saving of point coordinates aborted.\n")
        else:
            self.printlog("savepoints_fun(): Nothing selected, nothing saved.\n")
        self.update_figures_fun() # this should be called as often as possible

    def pointlist2matrix( self ):
        """ 
        extract the xy coordinates of points from self.pointlist
        output: np.matrix with 2 columns (x,y)
          NOTE: the output needs to be a np.matrix, not np.ndarray in order to have proper 
            indexing also for the case of just one point
        """
        
        # most pythonic: unzip, remove id, and zip again
        unzipped = list( zip(*self.pointlist) )
        xy = np.matrix( list( zip(unzipped[1],unzipped[2]) ) )
        
        # older methods
        # pointlistarr = np.matrix( self.pointlist )
        # point:(id,x,y)
        # xy = pointlistarr[:,[1,2]].astype(float)
        
        # old alternative version
        # xy = np.asmatrix( np.empty( (len(self.pointlist), 2 ) ) )
        # for i,point in enumerate(self.pointlist):
        #   xy[i,0] = float( point[1] ) # just in case, should be float already in pointlist
        #    xy[i,1] = float( point[2] )
        return xy

    def zoomtopoints_fun( self ):
        """
        Zoom view to the center of currently selected points with all selected points shown
        Loops over all selected windows
        """
        zoombuffer = 20 # the number of pixels between a point and figure edge
        #  the minimum size of the plot is thus 2*zoombuffer x 2*zoombuffer pixels
        selectedpoints = self.listbox_points.curselection()
        if len( selectedpoints ) > 0:
            xy = self.pointlist2matrix()[ selectedpoints, : ]
            curfigurelist = self.listbox_figures.curselection() # get the selected figures
            existingfigs = plt.get_fignums() # the list of existing figures
            if len( curfigurelist ) == 0:
                self.printlog("zoomtopoints(): No windows selected, nothing to zoom.\n")
            for curfigure in curfigurelist:
                curfigno = self.figurelist[ curfigure ][0]
                if curfigno in existingfigs:
                    fig_hypdata = self.figurelist[ curfigure ][ 2 ] 
                    xy_proj = world2image( self.figurelist[ curfigure ][ 4 ], xy )
                    minxy = np.min( xy_proj, axis=0 )
                    maxxy = np.max( xy_proj, axis=0 )
                    minx_i = minxy[0,0] - zoombuffer
                    miny_i = minxy[0,1] - zoombuffer
                    maxx_i = maxxy[0,0] + zoombuffer
                    maxy_i = maxxy[0,1] + zoombuffer
                    fig_hypdata.axes[0].set_xlim( ( minx_i, maxx_i ) )
                    fig_hypdata.axes[0].set_ylim( ( maxy_i, miny_i ) ) # y axes reversed, 0,0 in upper-left
                    set_display_square( fig_hypdata )
                else:
                    self.printlog("showpoints(): Figure " + str(curfigno) + " not open, not zooming it.\n")
        else:
            self.printlog("zoomtopoints(): No points selected, nothing to zoom to.\n")
        self.update_figures_fun() # this should be called as often as possible
        
    def showpoints_fun( self ):
        """
        Mark the locations of selected points in active raster windows.
        """
        selectedpoints = self.listbox_points.curselection()
        selectedcolor = self.pointcolor_string.get()
        if len( selectedpoints ) > 0:  
            curfigurelist = self.listbox_figures.curselection() # get the current figure
            existingfigs = plt.get_fignums() # the list of existing figures
            if len( curfigurelist ) == 0:
                self.printlog("showpoints(): No figures selected, cannot plot.\n")
            for curfigure in curfigurelist:
                curfigno = self.figurelist[ curfigure ][0]
                if curfigno in existingfigs:
                    fig_hypdata = self.figurelist[ curfigure ][ 2 ] 
                    xy_w = self.pointlist2matrix()[ selectedpoints, : ]
                    pointids = [ self.pointlist[i][0] for i in selectedpoints ]
                    xy = world2image( self.figurelist[ curfigure ][ 4 ] , xy_w )
                    for i,xy_row in enumerate(xy):
                        fig_hypdata.axes[0].plot( xy_row[0,0], xy_row[0,1], marker='x', c=selectedcolor )
                        fig_hypdata.axes[0].annotate( pointids[i], (xy_row[0,0]+1, xy_row[0,1]-1 ), color=selectedcolor )
                    fig_hypdata.canvas.draw()
                else:
                    self.printlog("showpoints(): Figure " + str(curfigno) + " not open, not plotting there.\n")
        else:
            self.printlog("showpoints(): No points selected, nothing to plot.\n")
        self.update_figures_fun() # this should be called as often as possible
            
    def pointsfromband_fun( self ):
        """
        Select pixels as points using band values as threshold.
        Opens a new window for dialogs.
        """
        # use the first select file in the list box
        if len( self.listbox_files.curselection() ) == 0:
            # I am not sure if this is even possible -- no selection
            self.printlog("pointsfromband_fun: no files selected. Not doing anything.\n")
        else:
            selection = self.listbox_files.curselection()[0]
            # PfBGUI wants just a sublist from the openfilelist here containing file handles of a single file
            GUI = PfBGUI( self.master, openfilelist=self.openfilelist[selection], 
                exportpointlist=self.pointlist )
    
    def loadshp_fun( self, idfieldname = None ):
        """
        loads a shapefile, or unloads it if loaded
        
        In:
        idfieldname: name of the field to use as ID for each polygon. If not given, automatic detection attempted or feature number used
        """
        if self.button_loadshp.cget('text') == "Load polygon from .shp":
            filename_shape = filedialog.askopenfilename(initialdir = self.foldername1, title = "Load vector shapefile",filetypes = (("shp files","*.shp"),("all files","*.*")))
            N_pts = 0 # the number of points in the final polygon. Used also to test if a polygon has been found
            if filename_shape != "":
                if len( self.polygonlist ) > 0:
                    self.printlog("loadshp_fun(): Some polygons already exist. Deleting.\n")
                    self.polygonlist = []
                    self.polygonIDlist = []
                sh_file = ogr.Open( filename_shape )
                if sh_file is not None:
                    # get some information and check for validity
                    N_layers = sh_file.GetLayerCount() 
                    self.printlog("Shapefile: " + str( N_layers ) + " layers" )
                    i_poly = 0 # the index counting the polygons in the file
                    # find the first layer with some features
                    for il in range(N_layers):
                        sh_layer = sh_file.GetLayerByIndex(il)
                        # print( sh_layer.GetExtent() )
                        sh_SpatialReference = sh_layer.GetSpatialRef()
                        self.printlog(", layer " + str(il) + ", "+ sh_layer.GetName() + ", has " + str(sh_layer.GetFeatureCount()) + " feature(s).\n" )
                        self.printlog("layer :" + sh_SpatialReference.ExportToProj4() +".\n")
                        # sh_f = sh_layer.GetFeature(0) #  The returned feature should be free with OGR_F_Destroy(). -- not done in Cookbook?
                        # get the names of all fields in the layer
                        #   also, identify a field to use as ID - use the first field with "id" in field name
                        fieldnames = []
                        ldefn = sh_layer.GetLayerDefn()
                        for iF in range(ldefn.GetFieldCount()):
                            fieldnames.append( ldefn.GetFieldDefn( iF ).name )
                            if idfieldname is None:
                                if fieldnames[-1].lower().find("id") > -1:
                                    idfieldname = fieldnames[-1]
                        self.printlog( "Fields " + ", ".join(fieldnames)+".\n"  )
                        if idfieldname is not None:
                            if idfieldname in fieldnames:
                                self.printlog( "Using field " + idfieldname + " for ID .\n")
                            else:
                                # this can happen if the idfieldname was given as input to the function
                                self.printlog( "Field " + idfieldname + " not found, not using for ID .\n")
                                idfieldname = None
                        sh_f = sh_layer.GetNextFeature()
                        featurenum = 1
                        while sh_f != None:
                            sh_g = sh_f.GetGeometryRef()
                            if idfieldname is not None:
                                field_ID = sh_f.GetFieldAsString(idfieldname)
                            else:
                                field_ID = str(il)+"_"+str(featurenum)
                            # try to make sure we have SpatialReference set. It seems to be lost sometimes
                            if sh_g.GetSpatialReference()==None and sh_SpatialReference!=None:
                                sh_g.AssignSpatialReference( sh_SpatialReference )
                            elif sh_g.GetSpatialReference()!=None:
                                sh_SpatialReference = sh_g.GetSpatialReference()                        
                            sh_g_type = sh_g.GetGeometryName() 
                            if sh_g_type == 'POLYGON':
                                # NOTE: some other geometries may also be potentially useful, see para 8.2.8 (page 66) of
                                #   http://portal.opengeospatial.org/files/?artifact_id=25355
                                # Points of polygon cannot be accessed directly, we need to get to the "ring" first
                                sh_g_ring = sh_g.GetGeometryRef(0)
                                # try to make sure we have SpatialReference set. It seems to be lost sometimes
                                if sh_g_ring.GetSpatialReference()==None and sh_SpatialReference!=None:
                                    sh_g_ring.AssignSpatialReference( sh_SpatialReference )
                                N_pts = sh_g_ring.GetPointCount()
                                if N_pts > 2:                                
                                    self.printlog( "Found geometry "+str(i_poly)+" of type " + sh_g_type + " with " + str(N_pts) + " points, ID="+field_ID+".\n")
                                    # plot the feature
                                    R = sh_g_ring.Clone() # store a clone so the file can be (hopefully) closed
                                    # self.printlog( "ring : "+ R.GetSpatialReference().ExportToProj4() +"\n" )
                                    self.polygonlist.append( R )
                                    self.polygonIDlist.append( field_ID )
                                    self.button_loadshp.configure( text = "Unload polygon")
                                    self.button_plotshp.configure( state=ACTIVE )
                                    self.button_zoomtoshp.configure( state=ACTIVE )
                                    self.button_pixelvalue.configure( state=DISABLED )
                                    self.button_zoomtopoints.configure( state=DISABLED )
                                    self.button_selectzoomedpoints.configure( state=DISABLED )
                                    self.button_showpoints.configure( state=DISABLED )
                                    self.button_addpoint.configure( state=DISABLED )
                                    self.button_deletepoints.configure( state=DISABLED )
                                    self.button_savepoints.configure( state=DISABLED )
                                    self.button_pixelvalue.configure( state=ACTIVE )
                                    self.button_plotspectra.configure( state=ACTIVE )
                                    self.button_analyzepoints.configure( state=ACTIVE )
                                    if len(self.figurelist) > 0:
                                        self.button_plotshp.configure( state=ACTIVE )
                                        self.button_zoomtoshp.configure( state=ACTIVE )
                                    # plot
                                    # if len(self.figurelist) > 0:
                                    #    self.plotshp_fun( i_poly )
                                    i_poly += 1
                                    # break
                            sh_f = sh_layer.GetNextFeature()
                            featurenum += 1
                        # end while loop over features
                    # end while loop  over layers
                if N_pts == 0:
                    self.printlog("loadshp_fun(): Could not load shapefile or no suitable features found.\n")
                # close file explicitly
                sh_file = None
            else:
                self.printlog("loadshp_fun(): Shapefile loading aborted.\n")
        else:
            # we were called to unload the shapefile
            self.button_loadshp.configure( text = "Load polygon from .shp")
            self.shapefilelist = [] # this should also close the shapefile
            self.button_plotshp.configure( state=DISABLED )
            self.button_zoomtoshp.configure( state=DISABLED )
            # activate point selection buttons, if appropriate
            self.listbox_points.configure( state=NORMAL )
            self.button_addpoint.configure( state=ACTIVE)
            self.polygonlist = []
            self.polygonIDlist = []
            if len(self.pointlist)>0:
                self.button_pixelvalue.configure( state=ACTIVE )
                if len(self.figurelist) > 0:
                    self.button_zoomtopoints.configure( state=ACTIVE )
                    self.button_selectzoomedpoints.configure( state=ACTIVE )
                    self.button_showpoints.configure( state=ACTIVE )
                    self.button_deletepoints.configure( state=ACTIVE )
                    self.button_savepoints.configure( state=ACTIVE )
                    self.button_plotshp.configure( state=DISABLED )
                    self.button_zoomtoshp.configure( state=DISABLED )
        self.update_figures_fun() # this should be called as often as possible
            
    def plotshp_fun( self, i_range=None ):
        """
        Plot the vector points in the ring self.polygonlist[i] in the active raster windows. Reprojects XY
        self.polygonlist is a "ring" of a polygon, of type osgeo.ogr.Geometry
        
        i: the index of the self.polygonlist elements to plot. If None, all will be plotted
        """
        # set up i_range. 
        if i_range is None:
            i_range = range( len(self.polygonlist) )
        else:
            # Check if i_range is iterable -- and make sure it is!
            try:
                temp = iter(i_range)
            except TypeError as te:
                i_range = [ i_range ]
        
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) == 0:
            self.printlog("plotshp_fun(): No figures selected, cannot plot.\n")
        for curfigure in curfigurelist:
            existingfigs = plt.get_fignums() # the list of existing figures
            curfigno = self.figurelist[ curfigure ][0]
            if curfigno in existingfigs:
                # each figurelist element should be a list: [ number, type, figurehandle, name, full_filename, DataIgnoreValue ]
                # the window is still open, use it
                fig_hypdata = self.figurelist[ curfigure ][2]
                hypfilename = self.figurelist[ curfigure ][4]
                # clear the plot of previous drawings
                while len( fig_hypdata.axes[0].get_lines() ) > 0:
                    fig_hypdata.axes[0].get_lines()[0].remove()
                self.printlog("plotshp_fun(): plotting polygons ")
                for i in i_range:
                    xy = shape2imagecoords( self.polygonlist[i], hypfilename )
                    # fig_hypdata.axes[0].plot( xy[:,0], xy[:,1], c='k' )
                    col = i % len(self.plotlinecolors)
                    fig_hypdata.axes[0].plot( xy[:,0], xy[:,1], c=self.plotlinecolors[col] )
                    self.printlog("#")
                fig_hypdata.canvas.draw()
                self.printlog("\n")
            else:
                self.printlog("plotvector(): Figure " +str(curfigno) + " not open, cannot plot.\n")
        self.update_figures_fun() # this is run as often as possible

    def zoomtoshp_fun( self ):
        """
        Zoom the currently active window around the shape (regardless if it's plotted or not)
        """
        zoombuffer = 20 # the number of pixels between a point and figure edge
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) == 0:
            self.printlog("showpoints(): No figures selected, cannot plot.\n")
        for curfigure in curfigurelist:
            existingfigs = plt.get_fignums() # the list of existing figures
            curfigno = self.figurelist[ curfigure ][0]
            if curfigno in existingfigs:
                # each figurelist element should be a list: [ number, type, figurehandle, name, full_filename, DataIgnoreValue ]
                # the window is still open, use it
                fig_hypdata = self.figurelist[ curfigure ][2]
                hypfilename = self.figurelist[ curfigure ][4]
                # get SpatialReference and GeoTransform
                SR_r, GT, startvalues = get_rastergeometry( hypfilename )  
                # initialize min and max values
                xmin_i = float('inf') 
                ymin_i = float('inf')
                xmax_i = -float('inf')
                ymax_i = -float('inf')
                # loop over all polygons 
                for R in self.polygonlist:
                    SR_v = R.GetSpatialReference()
                    vr_transform = osr.CoordinateTransformation( SR_v, SR_r )
                    # make a clone of the ring so the original would not be transformed
                    C = R.Clone()
                    C.Transform( vr_transform )
                    xy_w = np.array( C.GetPoints() ) # points in world coordinates
                    # transform to hyperspectral figure coordinates
                    D = GT[1]*GT[5] - GT[2]*GT[4]
                    xy = np.array([ (xy_w[:,0]*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - xy_w[:,1]*GT[2] ) / D - 0.5 ,
                        ( xy_w[:,1]*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - xy_w[:,0]*GT[4] ) / D - 0.5] ).transpose()
                    xmin_i = min( xmin_i, int( min( xy[:,0] ) - zoombuffer ) )
                    xmax_i = max( xmax_i, int( max( xy[:,0] ) + zoombuffer ) )
                    ymin_i = min( ymin_i, int( min( xy[:,1] ) - zoombuffer ) )
                    ymax_i = max( ymax_i, int( max( xy[:,1] ) + zoombuffer ) )
                fig_hypdata.axes[0].set_xlim( ( xmin_i, xmax_i ) )
                fig_hypdata.axes[0].set_ylim( ( ymax_i, ymin_i ) ) # y axes reversed, 0,0 in upper-left
                fig_hypdata.canvas.draw()
            else:
                self.printlog("plotvector(): Figure " +str(curfigno) + " not open, cannot plot.\n")
        self.update_figures_fun() # this is run as often as possible
    
    def loadfiles_fun( self ):
        """
        Load the hyperspectral files from the folder into the listbox. Does not open files.
        """
        # reset openfilelist and other environment as if no data file were loaded
        # self.openfilelist = []
        # self.button_loadfile.configure( background='SystemButtonFace' )
        # self.button_pixelvalue.configure( state=DISABLED )
        # self.button_zoomtopoints.configure( state=DISABLED )
        # self.button_selectzoomedpoints.configure( state=DISABLED )
        # self.button_showpoints.configure( state=DISABLED )

        # #clear the listbox before loading new data
        # self.listbox_files.delete( 0, END )

        filename1 =  filedialog.askopenfilename(initialdir=self.foldername1, title="Choose a file in a folder", filetypes=(("Envi hdr files","*.hdr"),("all files","*.*")))
        if filename1 != '':
            self.foldername1 = os.path.split(filename1)[0]
            filename1_nameonly = os.path.split(filename1)[1]
            self.printlog("setting foldername1 to "+self.foldername1 + ".\n")
            selectedfile = os.path.join( self.foldername1, filename1_nameonly )
            for file in os.listdir(self.foldername1): # file name (with extension, without path)
                filename = os.fsdecode(file) # I do not fully understand why this is necessary. Probably, can be omitted for most files
                if filename.lower().endswith(".hdr"):
                    self.openfilelist.append( [ os.path.join( self.foldername1, filename ), None, None, -1 ] )
            # populate the listbox
            # at the same time, get the list of full names of all files there
            fullnamelist = []
            self.listbox_files.delete(0,END)
            for item in self.openfilelist:
                # strip extension before storing
                self.listbox_files.insert( END, os.path.splitext(item[0])[0] )
                fullnamelist.append( item[0] )
            # select the item which was clicked (if it was a hdr file)
            if selectedfile in fullnamelist:
                self.listbox_files.selection_set( fullnamelist.index( selectedfile ) )

            self.button_displayfile.configure( state=ACTIVE )
            self.button_loadfile.configure( background='green' )
            self.button_deletefile.configure( state=ACTIVE )
            self.button_pointsfromband.configure( state=ACTIVE )
            if len(self.pointlist)>0:
                self.button_pixelvalue.configure( state=ACTIVE )
                self.button_updatepoints.configure( state=ACTIVE )
                self.button_plotspectra.configure( state=ACTIVE )
                self.button_analyzepoints.configure( state=ACTIVE )
                if len( self.figurelist ) > 0:
                    self.button_zoomtopoints.configure( state=ACTIVE )
                    self.button_selectzoomedpoints.configure( state=ACTIVE )
                    self.button_showpoints.configure( state=ACTIVE )
                    self.button_addpoint.configure( state=ACTIVE )
                    
    def deletefiles_fun( self):
        """
        Remove the selected file(s) from the listbox of files
        """
        selectedfileslist = self.listbox_files.curselection()
        if len( selectedfileslist ) == 0:
            self.printlog("deletefiles_fun(): No files selected, not modifying list.\n")
        else:
            for i in reversed(selectedfileslist):
                self.listbox_files.delete(i)
                self.openfilelist.pop(i)

    def displayfile_fun( self ):
        """
        load the selected files in listbox_files into new plots
        """
        
        if len( self.listbox_files.curselection() ) == 0:
            # I am not sure if this is even possible -- no selection
            self.printlog("displayfile_fun: nothing selected. Not doing anything.\n")
        else:
            for selection in self.listbox_files.curselection():
                # open the file if not open yet
                filename_short = self.listbox_files.get(selection) # file name without path or extension
                hypfilename = self.openfilelist[selection][0]
                
                # make sure the file is open
                self.openhypfile( selection )
                # use the existing file handles
                hypdata = self.openfilelist[selection][1]
                hypdata_map = self.openfilelist[selection][2]
                DIV = self.openfilelist[selection][3]
                stretchfactor = 0.95 # the default value
                stretchfactor = float( self.stretchfactor_string.get() )
                if stretchfactor > 1:
                    stretchfactor = 1
                elif stretchfactor < 0:
                    stretchfactor = 0.95
                self.printlog("displayfile_fun: using stretch = "+str(stretchfactor)+".\n")
                plotmode = self.plotmode_string.get()
                fig_hypdata = plot_hyperspectral( hypfilename, hypdata, hypdata_map, outputcommand=self.printlog, plotmode=plotmode, clip_up=stretchfactor )   
                # add figure to figurelist
                # self.figurelist = [] # each list element should be a list: [ number, type, figurehandle, name, filename_full, DataIgnoreValue ]
                ff = [ fig_hypdata.number, 'hyp', fig_hypdata, filename_short, hypfilename, DIV ]
                self.figurelist.append( ff )
                # add to list and highlight
                labelstring = str(ff[0])+' - '+ff[3]
                self.listbox_figures.insert( END, labelstring )
                self.listbox_figures.selection_set( END )
        
        self.update_figures_fun() # this should be called as often as possible
        
    def openhypfile( self, i_file ):
        """
        open file number i_file in self.listbox_files if not open yet
        """
        if self.openfilelist[i_file][1] is None :
            hypfilename = self.openfilelist[i_file][0]
            # hypfilename = os.path.join( self.foldername1, self.listbox_files.get( i_file ) )
            # open the file if not open yet. This only gives access to metadata.                
            hypdata = spectral.open_image( hypfilename )
            # open the file as memmap to get the actual hyperspectral data
            hypdata_map = hypdata.open_memmap()
            self.printlog("opening file "+hypfilename + "\n" )
            if hypdata.interleave == 1:
                self.printlog("Band interleaved (BIL).\n")
            else:
                self.printlog( hypfilename + " not BIL -- opening still as BIL -- will be slower.\n" ) 
                
            DIV = -1 # this means: do not use DIV
            if 'data ignore value' in hypdata.metadata:
                # actually, it seems Spectral converts DIV's to nan's
                #  so this is unnecessary
                #     but maybe not if the file contains ints?
                dtype = int( hypdata.metadata['data type'] )
                if dtype<4 or dtype > 11:
                    DIV = int( float(hypdata.metadata['data ignore value']) )
                else:
                    DIV = float( hypdata.metadata['data ignore value'] )                    
            self.openfilelist[ i_file ] = [ hypfilename, hypdata, hypdata_map, DIV ]

                
    def pixelvalue_fun( self ):
        """
        store the selected pixel values in a csv file
        double loop: over selected files and selected points
        """
        fn_name="pixelvalue_fun()"
        all_set = False
        selectedfileslist = self.listbox_files.curselection()
        N_files = len(selectedfileslist)
        if N_files == 0:
            self.printlog(fn_name+": No files selected, cannot continue.\n")
            # the for loop below will be skipped and no results produced
        # get the shape and dimensions of the requested area
        if self.button_loadshp.cget('text') == "Unload polygon":
            # the area is taken from polygon
            self.printlog(fn_name+": choosing by polygon.\n")
            N_points = len(self.polygonlist)
        else:
            # select pixels based on user selection
            areasize = float( self.areasize_string.get() )
            self.printlog("Area size:"+str(areasize) )
            if areasize == 0:
                self.printlog(fn_name+": could not get the value for area size. Aborting.\n")
                self.update_figures_fun() # this should be called as often as possible
                return
            else:
                areashape = self.areashape_string.get()
                areashape = areashape.split(" ")[1] # this returns either 'square,' or 'circle,'
                areaunit = self.areaunit_string.get() # either 'pixels' or 'meters'
                self.printlog( areaunit + ' ' + areashape + ".\n" )
                if areaunit == 'pixels':
                    areasize = int( round(areasize) )
                self.printlog( fn_name+": selecting a " + areashape + " " + str(areasize) + " " + areaunit + ".\n" )
                selectedpointlist = self.listbox_points.curselection()
                N_points = len( selectedpointlist )
                pointids = [ self.pointlist[i][0] for i in selectedpointlist ] 
                if N_points == 0:
                    self.printlog(fn_name+": No points selected, cannot continue.\n")
                    self.update_figures_fun() # this should be called as often as possible
                    return
        # ready to go!
        big_spectrumlist = [] # output: list of spectra
        wllist = [] # output: list of wavelengths. Will be ignored if usebands is set to True during processing
        filenamelist = [] # output: spectrum names
        big_Nlist = [] # output: the actual number of pixels averaged for each spectrum
        ProcessingFirstFile = True
        for i_file in selectedfileslist:
            self.openhypfile( i_file )
            hypfilename = self.openfilelist[i_file][0]
            shortfilename = os.path.split( hypfilename )[1]
            shortfilename = os.path.splitext( shortfilename )[0]
            filenamelist.append( shortfilename )
            hypdata = self.openfilelist[i_file][1]
            hypdata_map = self.openfilelist[i_file][2]
            DIV = self.openfilelist[i_file][3]
            
            if self.button_loadshp.cget('text') == "Unload polygon":                
                # retrieve spectra from polygons
                self.printlog(fn_name+": Loading spectrum data for " + str( N_points ) + " rings: " )
                wl,use_spectra = get_wavelength( hypfilename, hypdata )
                spectrumlist = []
                Nlist = []
                pointids = []
                for ring, featureID in zip(self.polygonlist, self.polygonIDlist):
                    pointids.append( featureID+":x"+str(round(ring.GetX(),3))+"y"+str(round(ring.GetY(),3)) ) # construct ID from coordinates
                    coordlist = points_from_shape( hypfilename, ring )
                    # if ProcessingFirstFile:
                    #     print(len(coordlist)
                    spectrum, N = avg_spectrum( hypfilename, coordlist, DIV, hypdata, hypdata_map )
                    spectrumlist.append( spectrum )
                    Nlist.append( N )
                    if ProcessingFirstFile:
                        self.printlog("(" + str(N) + ") ")
                if ProcessingFirstFile:
                    self.printlog(" points ... done\n")
            else:
                # spectra for points
                pointarray = self.pointlist2matrix()[ selectedpointlist, : ] # get numpy matrix of point coordinates
                spectrumlist, wl, Nlist = extract_spectrum( hypfilename, pointarray, areasize, areaunit, areashape, hypdata, hypdata_map )
            
            big_Nlist.append( Nlist )
            big_spectrumlist.append( spectrumlist )
            wllist.append( wl )
            ProcessingFirstFile = False
        self.printlog( fn_name+": read spectra from "+str(N_files)+" file(s), preparing to save. \n")
        # process the loaded spectra
        # find all possible wavelengths
        allwl = np.unique( np.concatenate( wllist ) )
        outmatrix = np.empty( (allwl.shape[0], N_points*N_files+1 ) )
        outmatrix[:,0] = allwl # wavelength as the first column
        legends = [ 'wl' ] # column headings in outmatrix
        c = 0 # the current column in outmatrix, set to zero (the column where wavelength is stored)
        for i_file in range( N_files ):
            wl = wllist[ i_file ]
            spectrumlist = big_spectrumlist[ i_file ]
            Nlist = big_Nlist[ i_file ]
            for i_point in range( N_points ):
                c += 1 # current column
                legends.append( filenamelist[ i_file ] + ":" + pointids[ i_point ]
                    + "(" + str( Nlist[ i_point] ) + ")" ) # legend: point_id with number of averaged spectra in parentheses
                for i_wl in range( allwl.shape[0] ):
                    #loop over wavelengths, ie. outmatrix rows
                    j = np.where( allwl[i_wl] == wl )[0] # find the location of the current wl in this specific hyp file
                    wl_exists = j.shape[0] != 0 # was this wavelength present in this file?
                    if wl_exists:
                        outmatrix[ i_wl , c ] = spectrumlist[ i_point ][ j[0] ]
                    else:
                        outmatrix[ i_wl , c ] = float('nan')
                    # end if
                # end loop over wavelengths
            # end loop over points
        # end loop over files
        if outmatrix.shape[0] > 0:
            self.printlog(fn_name+": opening file dialog ...")
            filename =  filedialog.asksaveasfilename(initialdir = self.foldername1, title = "Save extracted spectra",filetypes = (("csv files","*.csv"),("txt files","*.txt"),("all files","*.*")))
            if filename != '':
                delimiter = '\t'
                headerstring = ';'.join(legends)
                np.savetxt( filename, outmatrix, delimiter=delimiter, header=headerstring )
                self.printlog(" saved file "+filename+".\n")
            else:
                self.printlog(" saving of spectra aborted.\n")
        else:
            self.printlog(fn_name+": nothing to save, finishing\n")
        self.update_figures_fun() # this should be called as often as possible


    def plotspectra_fun( self ):
        """ Plot the spectra of selected points or loaded shapefile.
        Retrieves data from selected files with wavelength information.
        """
        functionname = "plotspectra(): "
        all_set = False
        selectedfileslist = self.listbox_files.curselection()
        N_files = len(selectedfileslist)
        if N_files == 0:
            self.printlog( functionname + "No files selected, cannot continue.\n" )
            # the for loop below will be skipped and no results produced
        else:
            # get the shape and dimensions of the requested area
            if self.button_loadshp.cget('text') == "Unload polygon":
                # the area is taken from polygon
                self.printlog( functionname + "Choosing by polygon.\n" )
                all_set = True
            else:
                # select pixels based on user selection
                areasize = float( self.areasize_string.get() )
                # self.printlog("Area size: "+str(areasize) )
                if areasize > 0:
                    areashape = self.areashape_string.get()
                    areashape = areashape.split(" ")[1] # this returns either 'square,' or 'circle,'
                    areaunit = self.areaunit_string.get() # either 'pixels' or 'meters'
                    # self.printlog( areaunit + ' ' + areashape + ".\n" )
                    if areaunit == 'pixels':
                        areasize = int( round(areasize) )
                    self.printlog( functionname + "Selecting a " + areashape + " " + str(areasize) + " " + areaunit + ".\n" )
                    selectedpointlist = self.listbox_points.curselection()
                    N_points = len( selectedpointlist )
                    if N_points == 0:
                        self.printlog( functionname + "No points selected, cannot continue.\n" )
                    else:
                        pointids = [ self.pointlist[i][0] for i in selectedpointlist ] 
                        all_set = True
                else:
                    self.printlog( functionname + "Could not get the value for area size. Aborting.\n" )
        if all_set:
            # the actual data retrieval and plotting part
            # open two plots: one for drawing against wavelengths, 
            # the other for files with no wavelenth information (plot against band number)
            # set both originally to None
            fig_bands = None
            fig_bands_axes = None
            fig_wl = None
            fig_wl_axes = None
            ProcessingFirstFile = True
            for i_style,i_file in enumerate(selectedfileslist):
                self.openhypfile( i_file )
                hypfilename = self.openfilelist[i_file][0]
                shortfilename = os.path.split( hypfilename )[1]
                shortfilename = os.path.splitext( shortfilename )[0]
                hypdata = self.openfilelist[i_file][1]
                hypdata_map = self.openfilelist[i_file][2]
                DIV = self.openfilelist[i_file][3]
                
                linestyle = self.plotlinestyles[ i_style % len( self.plotlinestyles ) ]
                if self.button_loadshp.cget('text') == "Unload polygon":
                    # retrieve spectra from polygons
                    self.printlog( functionname + "Loading spectrum data for " + str( len(self.polygonlist) ) + " rings: " )
                    wl,use_spectra = get_wavelength( hypfilename, hypdata )
                    spectrumlist = []
                    Nlist = []
                    pointids = []
                    for ring in self.polygonlist:
                        pointids.append( "x"+str(round(ring.GetX(),3))+"y"+str(round(ring.GetY(),3)) ) # construct ID from coordinates
                        coordlist = points_from_shape( hypfilename, ring )
                        spectrum, N = avg_spectrum( hypfilename, coordlist, DIV, hypdata, hypdata_map )
                        spectrumlist.append( spectrum )
                        Nlist.append( N )
                        if ProcessingFirstFile:
                            self.printlog("(" + str(N) + ") ")
                    if ProcessingFirstFile:
                        self.printlog(" points ... done\n")
                else:
                    pointarray = self.pointlist2matrix()[ selectedpointlist, : ] # get numpy matrix of point coordinates
                    spectrumlist, wl, Nlist = extract_spectrum( hypfilename, pointarray, areasize, areaunit, areashape, hypdata, hypdata_map )
                for i_point in range(len(spectrumlist)):
                    spectrum = spectrumlist[ i_point ]
                    N = Nlist[ i_point ]
                    linecolor = self.plotlinecolors[ i_point % len( self.plotlinecolors ) ]
                    if max( wl ) <= 0:
                        # bands are numbered with negative wavelengths
                        if fig_bands is None:
                            fig_bands = plt.figure()
                            fig_bands_axes = fig_bands.add_subplot(1,1,1)
                        fig_bands_axes.plot( -wl, spectrum, 
                            color=linecolor, linestyle=linestyle, label=shortfilename+','+pointids[i_point] )
                    else:
                        #plot against wavelength
                        if fig_wl is None:
                            fig_wl = plt.figure()
                            fig_wl_axes = fig_wl.add_subplot(1,1,1)
                        fig_wl_axes.plot( wl, spectrum, 
                            color=linecolor, linestyle=linestyle, label=shortfilename+','+pointids[i_point] )
                ProcessingFirstFile = False
            # Finishing touches
            if fig_bands != None:
                fig_bands_axes.set_title("Files without wavelength information")
                fig_bands_axes.set_xlabel("Band number")
                fig_bands_axes.legend()
                fig_bands.canvas.draw()
                fig_bands.show()
            if fig_wl != None:
                fig_wl_axes.set_title("Files with wavelength information")
                fig_wl_axes.set_xlabel("Wavelength")
                fig_wl_axes.legend()
                fig_wl.canvas.draw()
                fig_wl.show()

        self.update_figures_fun() # this should be called as often as possible
        
    def collectdata( self ):
        """ extract the values at point location from the data layers (spectral 
        bands of the different files) and arrange it into a data structure.
        returns: a data structure, list, containing all data values for the points
        
        only works for points, not shapes
        """
        
        """
        store the selected pixel values in a csv file
        double loop: over selected files and selected points
        """
        hypdatalist = [] # the big output data list, contains dicts for each data layer
        
        selectedfileslist = self.listbox_files.curselection()
        N_files = len(selectedfileslist)
        if N_files == 0:
            self.printlog("collectdata(): No files selected, cannot continue.\n")
            # the for loop below will be skipped and no results produced
        # get the shape and dimensions of the requested area

        # select pixels based on user selection
        areasize = float( self.areasize_string.get() )
        self.printlog("Area size:"+str(areasize) )
        if areasize == 0:
            self.printlog("collectdata(): could not get the value for area size. Aborting.\n")
            return None
        areashape = self.areashape_string.get()
        areashape = areashape.split(" ")[1] # this returns either 'square,' or 'circle,'
        areaunit = self.areaunit_string.get() # either 'pixels' or 'meters'
        self.printlog( areaunit + ' ' + areashape + ".\n" )
        if areaunit == 'pixels':
            areasize = int( round(areasize) )
        self.printlog("pixelvalue: selecting a " + areashape + " " + str(areasize) + " " + areaunit + ".\n" )
        
        selectedpointlist = self.listbox_points.curselection()
        N_points = len( selectedpointlist )
        points = [ self.pointlist[i] for i in selectedpointlist ]
        if N_points == 0:
            self.printlog(fn_name+": No points selected, cannot continue.\n")
            return None
        pointarray = self.pointlist2matrix()[ selectedpointlist, : ] # get numpy matrix of point coordinates
        
        # store the points as the 0th element of hypdatalist
        D = { "name":"pointlist" }
        D["filename"] = ""
        D["bandnumber"] = 0
        D["description"] = "point coordinates in global coordinates"
        D["foldername"] = ""
        D["wl"] = None
        D["areasize"] = areasize
        D["areaunit"] = areaunit
        D["areashape"] = areashape
        D["mapinfo"] = ""
        D["datalength"] = N_points
        D["data"] = points
        hypdatalist.append( D )
                
        counter = 0 # for printout
        for i_file in selectedfileslist:
            self.openhypfile( i_file )
            hypfilename = self.openfilelist[i_file][0]
            hypdata = self.openfilelist[i_file][1]
            hypdata_map = self.openfilelist[i_file][2]

            foldername = os.path.split( hypfilename )[0]
            shortfilename = os.path.split( hypfilename )[1]
            shortfilename = os.path.splitext( shortfilename )[0]
            
            wl_hyp, wl_found = get_wavelength( hypfilename, hypdata )
            N_bands = int( hypdata.metadata['bands'] )
            if 'band names' in hypdata.metadata:
                bandnames = hypdata.metadata['band names']
                # add wavelength as integer for each band name
                bandnames = [ str(int(i))+" "+j for i,j in zip(wl_hyp,bandnames) ]
            else:
                # name as wavelength: even if not given, negative integers are in wl_hyp
                bandnames = [ str(int(i)) for i in wl_hyp ]        
            if 'description' in hypdata.metadata:
                filedescription = hypdata.metadata["description"]
            else:
                filedescription = ""
            if 'map info' in hypdata.metadata:
                mapinfo = hypdata.metadata["map info"]
            else:
                mapinfo = ""
            # Note: DIVs are handled by default by extract_spectrum()

            spectrumlist, wl, Nlist = extract_spectrum( hypfilename, pointarray, areasize, areaunit, areashape, hypdata, hypdata_map )
            # the number of pixels is reported for each point and can vary, depending if the point is at the edge of the data or has NaNs.
            #   this is difficult to pass on into the data dictionary
            
            # split into data libraries: one library per band
            for i in range( N_bands ):
                D = { "name":bandnames[i] }
                D["filename"] = shortfilename
                D["bandnumber"] = i
                D["description"] = filedescription
                D["foldername"] = foldername
                D["wl"] = wl_hyp[i]
                D["areasize"] = areasize
                D["areaunit"] = areaunit
                D["areashape"] = areashape
                D["datalength"] = N_points
                D["data"] = [ spectrumlist[j][i] for j in range( len(spectrumlist)) ]
                hypdatalist.append( D )
                counter += 1
                
        self.printlog("collectdata(): collected "+ str(counter) + "data bands.\n")
        return hypdatalist
        
    def analyzepoints_fun( self ):
        """ Collect data for all poins in all bands of all open files and 
        start analysis tool in a separate GUI
        """
        hypdatalist = self.collectdata()
        GUI = AnalyzeGUI( self.master, hypdatalist, outputcommand=self.printlog )
            
    def update_figures_fun( self ):
        """ force the listbox_figures to correspond to actual opened figures, 
        i.e. detect manually deleted ones. Ignores figures not listed in figurelist 
        (created by other processes)
        """
        # plt.gcf().number # the number of the current figure, may open a figure if none are available
        existingfigs = plt.get_fignums() # the list of existing figures
        lastselection = self.listbox_figures.curselection() # get the current selection, try to restore it later
        if lastselection != ():
            # sth was selected
            lastselection_string = [ self.listbox_figures.get( i ) for i in lastselection ]
        else:
            # nothing was selected, use an empty string
            lastselection_string = list()
        selectiontohighlight = list() # the item to highlight at the end of script. Last (newest?) element if the previous not found    
        # clear listbox
        self.listbox_figures.delete(0,END)

        if len(self.figurelist) > 0:
            # we should have some figures
            # make a copy of the original figurelist and empty self.figurelist to refill from scratch
            copy_figurelist = self.figurelist
            self.figurelist = []
            currentfigure = 0 # the index in new, purged listbox
            
            for i,ff in enumerate(copy_figurelist):
                # self.figurelist = [] # each list element should be a list: [ number, type, handle, name ]self.figurelist = [] # each list element should be a list: [ number, type, figurehandle, name, full_filename ]
                if ff[0] in existingfigs:
                    # it has not been closed, update the listbox
                    labelstring = str(ff[0])+' - '+ff[3]
                    self.listbox_figures.insert( END, labelstring )
                    # copy it to new figurelist
                    self.figurelist.append( ff )
                    if labelstring in lastselection_string:
                        # highlight this item later
                        selectiontohighlight.append(currentfigure)
                    currentfigure += 1
        
            # see if we have anything left
            if len(self.figurelist) > 0:
                self.button_closefigure.configure( state=ACTIVE )
                self.button_clearfigure.configure( state=ACTIVE )
                self.button_addpoint.configure( state=ACTIVE )
                self.button_zoomout.configure( state=ACTIVE )
                self.button_zoomfull.configure( state=ACTIVE )
                self.button_zoomtofile.configure( state=ACTIVE )
                self.button_savezoomarea.configure( state=ACTIVE )
                if len( self.polygonlist ) > 0:
                    self.button_plotshp.configure( state=ACTIVE )
                    self.button_zoomtoshp.configure( state=ACTIVE )
                if ( len(self.openfilelist)>0 ) and ( len(self.pointlist)>0 ): # openfilelist should not be open anyway if a figure is open...
                    self.button_showpoints.configure( state=ACTIVE )
                    self.button_zoomtopoints.configure( state=ACTIVE )
                    self.button_selectzoomedpoints.configure( state=ACTIVE )

                for i in selectiontohighlight:
                    self.listbox_figures.selection_set( i )
            else:
                # all have been closed
                self.button_closefigure.configure( state=DISABLED )
                self.button_clearfigure.configure( state=DISABLED )
                self.button_plotshp.configure( state=DISABLED )
                self.button_zoomtoshp.configure( state=DISABLED )
                self.button_zoomout.configure( state=DISABLED )
                self.button_zoomfull.configure( state=DISABLED )
                self.button_zoomtofile.configure( state=DISABLED )
                self.button_zoomtopoints.configure( state=DISABLED )
                self.button_addpoint.configure( state=DISABLED )
                self.button_selectzoomedpoints.configure( state=DISABLED )
                self.button_savezoomarea.configure( state=DISABLED )
        else:
            # no figures to speak of
            self.button_closefigure.configure( state=DISABLED )
            self.button_clearfigure.configure( state=DISABLED )
            self.button_plotshp.configure( state=DISABLED )
            self.button_zoomtoshp.configure( state=DISABLED )
            self.button_zoomout.configure( state=DISABLED )
            self.button_zoomfull.configure( state=DISABLED )
            self.button_zoomtofile.configure( state=DISABLED )
            self.button_zoomtopoints.configure( state=DISABLED )
            self.button_addpoint.configure( state=DISABLED )
            self.button_selectzoomedpoints.configure( state=DISABLED )
            self.button_savezoomarea.configure( state=DISABLED )

        
    def clearfigure_fun( self ):
        """
        Clear the active raster plots of vector markings (points, polygons)
        """
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) == 0:
            self.printlog("clearfigure(): No figures selected, cannot plot.\n")
        for curfigure in curfigurelist:
            existingfigs = plt.get_fignums() # the list of existing figures
            curfigno = self.figurelist[ curfigure ][0]
            if curfigno in existingfigs:
                fig_hypdata = self.figurelist[ curfigure ][ 2 ] 
                while len( fig_hypdata.axes[0].get_lines() ) > 0:
                    fig_hypdata.axes[0].get_lines()[0].remove()
                # remove annotations
                for child in fig_hypdata.axes[0].get_children():
                    if isinstance(child, plt.Annotation):
                        child.remove()
                fig_hypdata.canvas.draw()
            else:
                self.printlog("clearfigure_fun(): Figure " + str(curfigno) + " not open, not doing anything.\n")
            
        self.update_figures_fun()
        
        
    def closefigure_fun( self ):
        """
        close the selected figure
        """
        # self.figurelist = [] # each list element should be a list: [ number, type, figure_handle, name, full_datafile ]
        curfigurelist = self.listbox_figures.curselection() # get the current figure
        if len( curfigurelist ) == 0:
            self.printlog("clearfigure(): No figures selected, nothing to close.\n")
        for curfigure in curfigurelist:
            existingfigs = plt.get_fignums() # the list of existing figures
            curfigno = self.figurelist[ curfigure ][0]
            if curfigno in existingfigs:
                plt.close( self.figurelist[ self.listbox_figures.curselection()[0] ][ 0 ] )
                # if the figure is already closed by user, nothing to do here
        self.update_figures_fun()
        
        
    def buttonquit_fun( self ):
        """
        function to end the misery
        note: the pyplot windows are not closed. Maybe, it would be nice to keep track of those to close them
        """
        set_hyperspectral_datafolder( self.foldername1 )
        self.master.destroy() # destruction of root required for program to continue



class AnalyzeGUI:
    """
    the GUI for simple analysis of data for selected points: plot one band against another
    """    

    def __init__(self, master, hypdatalist, outputcommand=None ):
        """ 
        hypdatalist: a list of dictionaries with data. First list element contains point coordinates
        Other dictionaries contain the values for each point from one band and metadata for the band
        """
        
        self.master = master # the master is likely tkroot
        self.hypdatalist = hypdatalist 
        self.outputcommand = outputcommand
        
        # Show the GUI in a Toplevel window instead of Root. 
        # This allows many such programs to be run independently in parallel.
        self.w = Toplevel( master )
        self.w.title("GUI for selecting pixels based on band values")
        
        # get all distinct file names
        self.filenames = list( np.unique( [ q["filename"] for q in hypdatalist[1:] ] ) )
        self.data1_set = False
        self.data2_set = False
        
        if outputcommand is None:
            self.textlog = ScrolledText( self.w, height=6 )
            self.textlog.pack( side='bottom' )
            self.outputcommand=printlog
        
        #just in case, put everythin in a frame, stuff can be added later if needed
        self.frame_buttons = Frame(self.w) 

        bw = 35 # button width
        self.label_1 = Label( self.frame_buttons, width=bw, text="Data #1: file and band")
        self.combo_file1 = ttk.Combobox( self.frame_buttons, values=self.filenames, name = "combo_file1" )
        if len(self.filenames) == 1:
            self.combo_file1.insert(0,self.filenames[0])
            bandnames = [ q["name"] for q in self.hypdatalist[1:] if q["filename"]==self.filenames[0] ]
            self.combo_band1 = ttk.Combobox( self.frame_buttons, values=bandnames, name = "combo_band1" )
            self.combo_band1.insert(0,"Select band")
        else:
            self.combo_file1.insert(0,"Select file")
            self.combo_band1 = ttk.Combobox( self.frame_buttons, values=[], name = "combo_band1" )
            self.combo_band1.insert(0,"No file selected")
        self.combo_file1['width'] = bw-5
        self.combo_band1['width'] = bw-5
        

        self.label_2 = Label( self.frame_buttons, width=bw, text="Data #2: file and band")
        self.combo_file2 = ttk.Combobox( self.frame_buttons, values=self.filenames, name = "combo_file2" )
        if len(self.filenames) == 1:
            self.combo_file2.insert(0,self.filenames[0])
            self.combo_band2 = ttk.Combobox( self.frame_buttons, values=bandnames, name = "combo_band2" )
            self.combo_band2.insert(0,"Select band")
        else:
            self.combo_file2.insert(0,"Select file")
            self.combo_band2 = ttk.Combobox( self.frame_buttons, values=[], name = "combo_band2" )
            self.combo_band2.insert(0,"No file selected")
        self.combo_file2['width'] = bw-5
        self.combo_band2['width'] = bw-5

        self.combo_file1.bind( "<<ComboboxSelected>>", self.comboprocess_file )
        self.combo_file2.bind( "<<ComboboxSelected>>", self.comboprocess_file )
        self.combo_band1.bind( "<<ComboboxSelected>>", self.comboprocess_band )
        self.combo_band2.bind( "<<ComboboxSelected>>", self.comboprocess_band )
        # event.widget._name in callback function gives the name
        self.button_plot = Button( self.frame_buttons, text='Plot', width=bw, command=self.plot_fun, state=DISABLED )
                
        
        self.label_1.pack( side='top' )
        self.combo_file1.pack( side='top' )
        self.combo_band1.pack( side='top' )
        self.label_2.pack( side='top' )
        self.combo_file2.pack( side='top' )
        self.combo_band2.pack( side='top' )
        self.button_plot.pack( side='top' )
        self.frame_buttons.pack( side='left' )

    def comboprocess_band( self, event=None, bandnumber=None ):
        """
        Catch clicks in the file selection comboboxes and updates the band list comboboxes
        event is the information created by the click
        bandnumber is an alternative way to call the function to create a fake click
        """

        if event is not None:
            bandnumber = int(event.widget._name[-1])
        if bandnumber == 1:
            self.data1_set = True
        elif bandnumber == 2:
            self.data2_set = True
        else:
            print(" --- comboprocess_band: sth strange happened "+ str(event) )
            return
        if self.data1_set and self.data2_set:
            self.button_plot.configure( state=ACTIVE )
        
    def comboprocess_file( self, event ):
        """
        Catch clicks in the file selection comboboxes and updates the file list comboboxes
        """
        
        if event.widget._name == "combo_file1":
            combo_band = self.combo_band1
            combo_file = self.combo_file1
            self.data1_set = False
        elif event.widget._name == "combo_file2":
            combo_band = self.combo_band2
            combo_file = self.combo_file2
            self.data2_set = False
        else:
            print(" --- comboprocess_file: sth strange happened:" + event.widget._name )
            return

        # combo_band['state'] = ACTIVE
        filename = combo_file.get()
        bandnames = [ q["name"] for q in self.hypdatalist[1:] if q["filename"]==filename  ]      
        combo_band.configure( values = bandnames )
        combo_band.delete(0,END)
        if len( bandnames) == 1:
            # fill the combobox with the only band and simulate a selection event
            combo_band.insert(0,bandnames[0])
            self.comboprocess_band( bandnumber=int(event.widget._name[-1]) )
        else:
            combo_band.insert(0,"Select band")
            self.button_plot.configure( state=DISABLED )
        
    def get_data( self ):
        """
        Returns the selected x1, x2 data dictionaries
        """
        filename1 = self.combo_file1.get()
        bandname1 = self.combo_band1.get()
        filename2 = self.combo_file2.get()
        bandname2 = self.combo_band2.get()
        x1D = [ q for q in self.hypdatalist[1:] 
            if q["filename"]==filename1 and q["name"]==bandname1 ]
        x2D = [ q for q in self.hypdatalist[1:] 
            if q["filename"]==filename2 and q["name"]==bandname2 ]
        
        if len( x1D ) > 1:
            self.outputcommand("non-unique data selection for #1\n")
        if len( x2D ) > 1:
            self.outputcommand("non-unique data selection for #2\n")
        if len(x1D)==0 or len(x2D)==0:
            self.outputcommand("zero-length data vector, lengths:"+
                str(len(x1D))+","+str(len(x2D))+"\n")
            return None,None
        return x1D[0], x2D[0]
        
    def plot_fun( self ):
        """
        The most basic analysis, plot data
        """
        x1D,x2D = self.get_data()
        x1 = x1D["data"]
        x2 = x2D["data"]
        fig_plot = plt.figure() # later, maybe make use of the handle
        fig_plot_axes = fig_plot.add_subplot(1,1,1)
        fig_plot_axes.plot( x1, x2, "x" )
        fig_plot_axes.set_xlabel(x1D["name"]+" "+x1D["filename"])
        fig_plot_axes.set_ylabel(x2D["name"]+" "+x2D["filename"])
        fig_plot.show()

    def printlog( self , text ):
        """
        Output to log window. Note: no newline added beteen inputs.
        text need not be a string, will be converted when printing.
        """
        self.textlog.insert( END, str(text) )
        self.textlog.yview( END )
        self.master.update_idletasks()



if __name__ == '__main__':
    root = Tk()
    GUI = pixelGUI( root )
    root.withdraw()
    root.mainloop()
