from tkinter import *
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import threading
import time
import os

from spectralinvariant.hypdatatools_img import *
from spectralinvariant.hypdatatools_gdal import *

# GUI for selecting pixel centers from image as a set of points
        
class PfBGUI:
    """
    the GUI for selecting pixel centers from image as a set of points
    """    
    
    hypdatadir="" # location of hyperspectral data, initial suggestion in file dialog

    def __init__(self, master, openfilelist=None, exportpointlist=None ):
        """ 
        signalPfB: variable to signal the caller that we are finished, type DoubleVar() 
        openfilelist: list of the loaded hyperspectral file name and handles: [ filename filehandle datahandle ]
            intially, when loading the hyperspectral handles are set to None; they are assigned when file is opened for e.g. plotting
        exportpointlist: output list of tuples (id,x,y), where x,y are point (pixel centre) coordinates in the global projected system
            the pointlist created here is appended to exportpointlist
        if called with an exportpointlist, upon exit emits the signal master.<<PfBGUI_exit>>
        """
        
        self.master = master # the master is likely tkroot
        self.openfilelist = openfilelist
        self.exportpointlist = exportpointlist
        
        # Show the GUI in a Toplevel window instead of Root. 
        # This allows many such programs to be run independently in parallel.
        self.w = Toplevel( master )
        self.w.title("GUI for selecting pixels based on band values")
        
        self.bandnames = [] # list of band names as strings      
        self.coordlist = [ [],[] ] # list of coordinate points as produced by np.nonzero(), i.e., list of two lists, containing x and y coordinates, respectively
        # NOTE: these are image coordinates (line,pixel), i.e., (y,x)
        #  will be upended to exportpointlist on exit
        self.DIV = StringVar() # Data Ignore Value
        self.DIV.set("")
        self.fig_hypdata = None # figure handle for the image of th selected band
        self.xlim = [] # plotting limits for fig_hypdata: these will be retained during band and mask changes
        self.ylim = []
        self.figmask = None # handle for the mask plt.imshow object
        
        self.band = StringVar() # tkinter string to set and get the value in band optionmenu
        
        self.textlog = ScrolledText( self.w, height=6 )

        #just in case, put everythin in a frame, stuff can be added later if needed
        self.frame_buttons = Frame(self.w) 

        bw = 35 # button width
        self.button_quit = Button( self.frame_buttons, text='´Done', width=bw, command=self.buttondone )
        self.button_datafile = Button( self.frame_buttons, text='Load datafile', width=bw, command=self.datafile )
        self.button_savepoints = Button( self.frame_buttons, text="save points in file", width=bw, command=self.savepoints )
        
        self.combo_band = ttk.Combobox( self.frame_buttons, textvariable=self.band, values=["select band"] )
        self.combo_band.bind( "<<ComboboxSelected>>", self.selectband )
        self.combo_band['width'] = bw-5
        self.combo_band['state'] = DISABLED
        
        self.button_plotband = Button( self.frame_buttons, text='Plot band', width=bw, command=self.plotband, state=DISABLED )
        self.label_min = Label( self.frame_buttons, width=bw, text="Lower value:")
        self.minvaluestring = StringVar()
        self.minvaluestring.set("-")
        self.entry_minvalue = Entry( self.frame_buttons, textvariable=self.minvaluestring )
        self.label_max = Label( self.frame_buttons, width=bw, text="Upper value:")
        self.maxvaluestring = StringVar()
        self.maxvaluestring.set("-")
        self.entry_maxvalue = Entry( self.frame_buttons, textvariable=self.maxvaluestring )
        self.button_applyrange = Button( self.frame_buttons, text='Pick points', width=bw, command=self.applyrange, state=DISABLED )
        
        self.label_DIV = Label( self.frame_buttons, width=bw, text="Data Ignore Value:" )
        self.entry_DIV = Entry( self.frame_buttons, width=bw, textvariable=self.DIV )

        self.label_id = Label( self.frame_buttons, width=bw, text="ID string for points:")
        self.point_id = StringVar()
        self.point_id.set("THRSHLD")
        self.entry_id = Entry( self.frame_buttons, width=bw, textvariable=self.point_id )
        
        self.label_N = Label( self.frame_buttons, width=bw, text="Points: 0")

        self.textlog.pack( side='bottom' )
        
        if self.openfilelist is None:
            self.button_datafile.pack( side='top' )
        else:
            # load the data file directly
            # assume that at least file name is given
            if self.openfilelist[1] is None or self.openfilelist[2] is None:
                # the data files are not opened yet
                self.load_hypdata()
            self.fill_bandnames()

        self.combo_band.pack( side='top' )
        self.button_plotband.pack( side='top' )
        self.label_min.pack( side='top' )
        self.entry_minvalue.pack( side='top' )
        self.label_max.pack( side='top' )
        self.entry_maxvalue.pack( side='top' )
        self.button_applyrange.pack( side='top' )
        self.button_savepoints.pack( side='top' )
        self.label_N.pack( side='top' )
        self.label_DIV.pack( side='top' )
        self.entry_DIV.pack( side='top' )
        self.label_id.pack( side='top' )
        self.entry_id.pack( side='top' )
        
        
        self.button_quit.pack( side='bottom' )

        self.frame_buttons.pack( side='left' )

        
    def datafile( self ):
        """
        get data file name and load metadata
        """
        
        filename1 =  filedialog.askopenfilename(initialdir = self.hypdatadir, title = "Choose hyperspectal data file", filetypes = (("ENVI header files","*.hdr"),("all files","*.*")))
        
        if filename1 != "" :
            self.openfilelist = [ filename1, None, None ]
            self.load_hypdata() 
            self.fill_bandnames()
            
        shortfilename = os.path.split( filename1 )[1]
        shortfilename = os.path.splitext( shortfilename )[0]
        self.button_datafile.configure( text="File: "+shortfilename )

        
    def load_hypdata( self ):
        """
        load the file handles into openfilelist and check for wavelength data
        """
        # open the data file -- reads only metadata
        hypdata = spectral.open_image( self.openfilelist[0] )
        # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
        # print(hypdata.metadata.keys())

        if hypdata.interleave == 1:
            self.printlog( self.openfilelist[0] + " Band interleaved (BIL) \n")
        else:
            self.printlog( self.openfilelist[0] + " not BIL -- opening still as BIL -- will be slower \n" )
        hypdata_map = hypdata.open_memmap()
        self.printlog("data file dimensions " + ", ".join( [ str(i) for i in hypdata_map.shape ] ) + "\n" ) #shape[0]==lines, shape[1]==pixels, shape[2]==bands
        
        # save the handles to the openfilelist
        self.openfilelist = [ self.openfilelist[0], hypdata, hypdata_map ]     
        
    def fill_bandnames( self ):
        """
        Retrieve band names from the datafile metadata and fill the OptionMenu 
        Sets self.DIV Entry (Data Ignore Value)
        """
        hypfilename = self.openfilelist[0]
        hypdata = self.openfilelist[1]
        
        self.bandnames = []
        self.wl_hyp, wl_found = get_wavelength( hypfilename, hypdata )

        if 'band names' in hypdata.metadata:
            self.bandnames = hypdata.metadata['band names']
            # add wavelength as integer for each band name
            self.bandnames = [ str(int(i))+" "+j for i,j in zip(self.wl_hyp,self.bandnames) ]
        else:
            # name as wavelength: even if not given, negative integers are in wl_hyp
            self.bandnames = [ str(int(i)) for i in self.wl_hyp ]        
        
        DIV = get_DIV( hypfilename, hypdata )
        if DIV is not None:
            self.DIV.set( str(DIV) )
            self.entry_DIV.delete( 0, END )
            self.entry_DIV.insert( 0, str(DIV) )
        # fill the band name ComboBox
        self.combo_band['state'] = ACTIVE
        self.combo_band.configure( values = self.bandnames )
        self.combo_band.delete( 0, END )
        self.combo_band.insert( 0, self.bandnames[0] )
        self.selectband()
                    
        self.button_plotband.configure( state=ACTIVE )
        self.button_applyrange.configure( state=ACTIVE  )
                
    def selectband( self, event=None ):
        """
        Activate a selection in the band selection ComboBox 
        and load pixel value ranges
        """
        choice = self.combo_band.get()
        bandnumber = self.bandnames.index(choice)
        bandimage = self.openfilelist[2][ :,:,bandnumber]
        if self.DIV.get() != "":
            DIV = float( self.DIV.get() )
        else:
            DIV = None
        i_data = np.nonzero( bandimage != DIV )
        minvalue = np.min( bandimage[i_data] )
        maxvalue = np.max( bandimage[i_data] )
        self.entry_minvalue.delete( 0, END )
        # to avoid digitization errors, we must always know if the field contains the actual min and max
        self.entry_minvalue.insert(0, "min="+str(minvalue) ) #prefix with min to indicate that we have the histogram minimum
        self.entry_maxvalue.delete( 0, END )
        self.entry_maxvalue.insert(0, "max="+str(maxvalue) )

                
    def plotband(self):
        """ 
        plot the band using imshow
        """
        bandnumber = self.bandnames.index(self.band.get())
        self.printlog( "Plotting band " + self.band.get() +" [" + str(bandnumber)+"]\n" )
        
        hypfilename = self.openfilelist[0]
        hypdata = self.openfilelist[1]
        hypdata_map = self.openfilelist[2]

        self.xlim = None
        self.ylim = None        
        if self.fig_hypdata is not None:
            if self.fig_hypdata.number in plt.get_fignums():
                # save the current view if the window exists
                self.xlim = self.fig_hypdata.axes[0].get_xlim()
                self.ylim = self.fig_hypdata.axes[0].get_ylim()

        if self.figmask is not None:
            # remove the old mask
            self.figmask.remove() 
            self.figmask = None

        self.fig_hypdata = plot_singleband( hypfilename, hypdata, hypdata_map, bandnumber, 
            fig_hypdata=self.fig_hypdata, outputcommand=self.printlog )
            
        if self.xlim is not None:
            # restore the previous view
            self.fig_hypdata.axes[0].set_xlim( self.xlim )
            self.fig_hypdata.axes[0].set_ylim( self.ylim )
            self.fig_hypdata.canvas.draw()
        
    def savepoints( self ):
        """
        save the created points (i.e., their coordinates, not spectra) to a text file.
        """
        if len( self.coordlist[0] ) > 0:
            filename =  filedialog.asksaveasfilename(initialdir = self.hypdatadir, title = "Save point coordinates (X,Y)",filetypes = (("txt files","*.txt"),("csv files","*.csv"),("all files","*.*")))
            if filename != '':
                with open(filename,'w') as file:
                    file.write("id,x,y\n")
                    pointlist = self.get_pointlist()
                    for point in pointlist:
                        pointstring = point[0] + "," + str(point[1]) + ',' + str(point[2]) + '\n'
                        file.write( pointstring )
                self.printlog("Point coordinates saved to "+ filename +".\n")
            else:
                self.printlog("Saving of point coordinates aborted.\n")
        else:
            self.printlog("savepoints(): No points, nothing saved.\n")
            
        
    def applyrange( self ):
        """
        Create the points
        """
 
        hypfilename = self.openfilelist[0]
        hypdata_map = self.openfilelist[2]
        bandnumber = self.bandnames.index(self.band.get())
        bandimage = hypdata_map[ :,:,bandnumber]
        if self.DIV.get() != "":
            DIV = float( self.DIV.get() )
        else:
            DIV = None
        i_data = np.nonzero( bandimage != DIV )
        
        if self.minvaluestring.get()[0] == "m":
            # the entry field contains unmodified histogram minimum
            minvalue = np.min( bandimage[i_data] )
        else:
            minvalue = float( self.minvaluestring.get() )
        
        if self.maxvaluestring.get()[0] == "m": 
            # the entry field contains unmodified histogram maximum
            maxvalue = np.max( bandimage[i_data] )
        else:
            maxvalue = float( self.maxvaluestring.get() )
        
        self.coordlist = np.nonzero( np.logical_and( np.logical_and (
            bandimage>=minvalue, bandimage<=maxvalue), bandimage != DIV ) )
        N = len( self.coordlist[0] )
        self.label_N.configure( text = "Points: " + str(N) )
        
        # plot the mask only if the band image exists
        if self.fig_hypdata is not None:
            if self.fig_hypdata.number in plt.get_fignums():
                # just in case: save the current view
                self.xlim = self.fig_hypdata.axes[0].get_xlim()
                self.ylim = self.fig_hypdata.axes[0].get_ylim()
                mask = np.zeros_like( bandimage, dtype=float ) # NaN's don't work with integers
                mask[ self.coordlist ] = np.NaN # NaN plots as transparent
                
                if self.figmask is not None:
                    # remove the old mask
                    self.figmask.remove() 
                self.figmask = self.fig_hypdata.axes[0].imshow( mask, interpolation='nearest')

                # restore the previous view
                self.fig_hypdata.axes[0].set_xlim( self.xlim )
                self.fig_hypdata.axes[0].set_ylim( self.ylim )
                    
                self.fig_hypdata.canvas.draw()

    def buttondone(self ):
        """
        function to end the misery and return the points
        """
        if self.fig_hypdata is not None:
            plt.close( self.fig_hypdata )
        if self.exportpointlist is not None and len(self.coordlist[0]) > 0:
            pointlist = self.get_pointlist()
            self.exportpointlist += pointlist
            self.master.event_generate("<<PfBGUI_exit>>", when="tail")
            self.w.destroy() # if we were called to create points, don't destroy root, computation will continue elsewhere
        else:
            self.master.destroy() # destruction of root will exit the main loop and allow the program to exit
            
    def get_pointlist( self ):
        """
        outputs pointlist, a list of tuples (id,x,y) using ID and self.coordlist (y_loc, x_loc)
        x,y are in global coordinates
        """
        id = self.point_id.get()
        IDlist = [id] * len( self.coordlist[0] )
        pointmatrix_local = np.matrix(self.coordlist).transpose()[:,(1,0)] # swap x & y
        pointmatrix_global = image2world( self.openfilelist[0], pointmatrix_local )
        pointlist =  [ (id,Q[0],Q[1]) for id,Q in zip(IDlist, pointmatrix_global.tolist() ) ] 
        return pointlist

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
    GUI = PfBGUI( root )
    root.withdraw()
    root.mainloop()


    