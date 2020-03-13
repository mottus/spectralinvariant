from tkinter import *
from tkinter import filedialog, ttk
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy import stats
import threading
import os
import time
import sys
AIROBEST_dir = 'hyperspectral\\AIROBEST'
if not AIROBEST_dir in sys.path:
    sys.path.append(AIROBEST_dir)
# sys.path.append("C:\\Users\\MMATTIM\\OneDrive - Teknologian Tutkimuskeskus VTT\\koodid\\python\\hyperspectral\\AIROBEST")
# sys.path.append("hyperspectral\\AIROBEST")

from hypdatatools_algorithms import *
from spectralinvariants import *

# GUI for running p-computations
    
class p_thread( threading.Thread ):
    """
    the thread which will run the process
    """
    def __init__( self, tkroot, filename1, refspecno, wl_p, filename2, filename3, 
        hypdata, hypdata_map, progressvar, refspec, refspecname ):
        """
        the inputs are
        tkroot: handle for the root window for signaling
        the rest are forwarded to p_processing()
            filename1 (str): reference spectrum file name (incl full directory)
            refspecno (int): spectrum in filename1 to use
            wl_p ([int]): index of wavelengths used in computations 
            filename2 (str): the hyperspectraldata file which is used as input
            filename3 (str): the name of the data file to create
            optional inputs:
            tkroot : tkinter root window handle for signaling progress
            file2_handle=None : the spectral pyhton file handle if file is already open (for metadata)
            file2_datahandle=None : the spectral pyhton handle for hyperspectral data if file is already open
                filename2 is not reopened if the file handle exists (data handle is not checked)
            progressvar: NEEDS TO BE DoubleVar (of tkinter heritage) -- the variable used to mediate processing progress with a value  between 0 and 0.
                progressvar is also used to signal breaks by setting it to -1
            refspec=None: the actual reference spectrum data
                if not None, filename1 and refspecno will not be used 
            refspecname='RefenceSpectrum': name of the refrence spectrum given in refspec
        """
        threading.Thread.__init__(self) # this call is apparently mentioned as a requirement in some tutorial
        self.tkroot = tkroot
        self.filename1 = filename1
        self.refspecno = refspecno
        self.wl_p = wl_p
        self.filename2 = filename2
        self.filename3 = filename3
        self.hypdata = hypdata
        self.hypdata_map = hypdata_map
        self.progressvar = progressvar
        self.refspec = refspec
        self.refspecname = refspecname
    
    def run(self):
        """
        a wrapper for running p_processing in a separate thread
        """
        # do the thing
        p_processing( self.filename1, self.refspecno, self.wl_p, self.filename2, self.filename3, self.tkroot, self.hypdata, self.hypdata_map, 
            progressvar=self.progressvar, refspec=self.refspec, refspecname=self.refspecname )
        # signal that we have finished
        self.tkroot.event_generate("<<thread_end>>", when="tail")

class pGUI:
    """
    the GUI for creating the thread which will do the calculation
    """
    
    def __init__(self, master):
        
        self.progressvar_p = DoubleVar() # for signaling breaks and progress
        self.progressvar_p.set(0)
        
        self.filename1 = "" # file for reference spectra
        self.filename2 = "" # hyperspectral data file
        self.refspecname = "" # name of the loaded reference spectrum
        # the following two numpy arrays are defined in buttonspectrumfile (and reloaded in buttonloadref)
        #  (python does not require them to be declared here, it's only for clarity)
        self.refspec = None
        self.wl_spec = None
        self.hypdata = None # the spectral.io.bsqfile.BsqFile object
        self.hypdata_map = None # the memmap of hypdata. I am not sure but I think it's best to open hypdata only once and retain the handle
        
        # by default, load PROSPECT transformed reference spectrum. A new spectrum from a file can be loaded later
        # note: if a reference spectrum file is loaded, this information may not be used anymore
        self.refspec = spectralinvariants.referencealbedo_transformed()
        self.refspecname = "Transformed_PROSPECT"
        self.wl_spec = spectralinvariants.reference_wavelengths()
        self.use_refspec_file = False # whether to load reference spectra from file
        
        self.wl_hyp = np.array([]) # wavelengths of hyperspectral, hopefully in nm
        self.refspectranames = [] # list of loaded reference spectra names
        self.openfilelist = [] # list of loaded file names and handles
            # each element is a list [ filename filehandle datahandle ]
            # intially, when loading the hyperspectral handles are set to None; they are assigned when file is opened for e.g. plotting
            # the list will contain only one element as only one file is opened in this program

        
        self.specdatadir="C:\\data\\koodid\\idl\\" # location of (leaf) sppectrum files, initial suggestion in file dialog
        self.hypdatadir="D:\\mmattim\\wrk\\hyytiala-D\\" # location of hyperspectral data, initial suggestion in file dialog
        
        self.fig_refspec = None # handle for reference spectrum figures
        self.ax_refspec = None # handle for the axes in refspec
        self.fig_hypdata = None # handle for hyperspectral image
    
        self.fig_spectrum = None # handle for pixel spetrum figures
        self.ax_spectrum = None # handle for the axes in fig_spectrum
        self.fig_pplot = None # handle BRF/w vs. BRF figure
        self.ax_pplot = None # handle for the axes in fig_pplot
            
        self.reference_loaded = False # flag
        self.hypdata_loaded = False # flag
        
        self.hypdata_ciglock = False # cig lock for fig_hypdata so only one function can catch clicks from pyplot
        self.catch_cid = -1 # connection id for pyplot mouse click
        
        self.master = master
        master.title("GUI for calculating p")
        self.hypdata_ciglock = False
        self.catch_cid = -1 # connection id for pyplot mouse click
        self.fig_pplot = None
        self.fig_spectrum = None 
        self.fig_refspec = None
        self.fig_hypdata = None
        
        # add a listbox in a frame with a scrollbar for spectra names
        self.frame_listbox = Frame(master)
        self.scrollbar = Scrollbar(self.frame_listbox, orient='vertical') 
        self.listbox = Listbox(self.frame_listbox, exportselection=False, yscrollcommand=self.scrollbar.set )
        self.scrollbar['command'] = self.listbox.yview
        self.button_plotref = Button(self.frame_listbox, text="Plot", width=20, command=self.buttonplotref )
        self.button_plotref.pack( side='bottom' )
        self.scrollbar.pack( side='right', fill='y' )
        self.listbox.pack( side='top' )
        self.listbox.insert( END , self.refspecname )
        
        # add a listbox in a frame with a scrollbar for wavelengths
        self.frame_listbox_wl = Frame(master)
        self.scrollbar_wl = Scrollbar(self.frame_listbox_wl, orient='vertical') 
        self.listbox_wl = Listbox(self.frame_listbox_wl, exportselection=False, selectmode='extended', yscrollcommand=self.scrollbar_wl.set )
        self.scrollbar_wl['command'] = self.listbox_wl.yview
        self.scrollbar_wl.pack(side="right", fill="y")
        self.listbox_wl.pack()
        
        # a frame for buttons to load stuff etc.
        bw = 25 # button width
        self.frame_buttons = Frame(master)
        self.button_quit = Button( self.frame_buttons, text='Quit', width=bw, command=self.buttonquit )
        self.button_datafile = Button( self.frame_buttons, text='set datafile', width=bw, command=self.buttondatafile )
        self.button_spectrumfile = Button( self.frame_buttons, text='set spectrum file', width=bw, command=self.buttonspectrumfile )
        self.button_p = Button( self.frame_buttons, text='p for pixel', width=bw, command=self.p_pixel, state=DISABLED )
        self.button_clearpoints = Button( self.frame_buttons, text='Clear points', width=bw, command=self.clearpoints, state=DISABLED )
        self.button_run = Button( self.frame_buttons, text='Run', width=bw, command=self.buttonrun, state=DISABLED )
        self.progressbar = ttk.Progressbar( self.frame_buttons, orient='horizontal', maximum=1, value=0, variable=self.progressvar_p, mode='determinate' )
        
        # self.button_datafile.pack(pady=20, padx = 20)
        self.button_datafile.pack( side='top' )
        self.button_spectrumfile.pack( side='top' )
        self.button_p.pack( side='top' )
        self.button_clearpoints.pack( side='top' )
        self.button_run.pack( side='top' )
        self.button_quit.pack( side='top' )
        self.progressbar.pack( side='bottom', fill=X )

        self.frame_buttons.pack(side="left")
        self.frame_listbox_wl.pack(side="right")
        self.frame_listbox.pack(side="right")

        master.bind("<<thread_end>>", self.thread_end )
        
    def thread_end(self,*args):
        """
        function to restore stuff for potential next processing
        """
        self.button_quit.configure( state=ACTIVE )
        self.button_datafile.configure( state=ACTIVE )
        self.button_spectrumfile.configure( state=ACTIVE )
        self.button_run.configure( text='Run' )
        if self.reference_loaded and self.hypdata_loaded:
            self.button_run.configure( state=ACTIVE )
            self.button_p.configure( state=ACTIVE )
        print("Thread exit caught")

    def buttonquit(self):
        """
        function to end the misery
        note: the pyplot windows are not closed. Maybe, it would be nice to keep track of those to close them
        this is of course irrelevant when run from a command line and not interactive shell
        """
        self.master.destroy() # destruction of root required for program to continue
    
    def buttonspectrumfile(self):
        """
        get reference spectrum file name and read the names of spectra in it
        """
        self.filename1 =  filedialog.askopenfilename(initialdir = self.specdatadir, title = "leaf spectrum file",
                filetypes = (("csv files","*.csv"),("text files","*.txt"),("all files","*.*")))
        if self.filename1  != "":
            # read first line assuming it contain tab-delimeted column headings
            with open(self.filename1) as f:
                self.refspectranames = f.readline().strip().split('\t')
                
            if self.refspectranames[0][0] == "#":
                # remove the comment symbol
                self.refspectranames[0] = self.refspectranames[0][1:] # this may retain an unwanted underscore, but let it be
            else:
                # names not given on first line
                self.refspectranames = [str(i) for i in range(len(self.refspectranames)) ]

            #  the first (zeroeth) element in self.refspectranames is "wl"
            # clear listbox and load the names of all available spectra
            self.listbox.delete( 0, END )
            for item in self.refspectranames[1:]:
                self.listbox.insert(END,item)
                
            # which spectra should we use by default?
            if "PROSPECT" in self.refspectranames:
                # load PROSPECT automatically
                i_P = self.refspectranames.index("PROSPECT")
                self.reference_loaded = True
                # load and plot reference 
                leafspectra = np.loadtxt(self.filename1)
                self.wl_spec = leafspectra[:,0]
                self.listbox.selection_set( i_P-1 ) # first column is "wl"
              
                if self.hypdata_loaded:
                    self.button_run.configure( state=ACTIVE )
                    self.button_p.configure( state=ACTIVE )
            else:
                # choose the first element
                self.listbox.selection_set(0)
                self.wl_spec = leafspectra[:,0]
                
            # self.button_plotref.configure( state=ACTIVE )
            self.use_refspec_file = True
            self.button_spectrumfile.configure( background='green' )
        else:
            # load the default PROSPECT transformed reference spectrum. A new spectrum from a file can be loaded later
            # note: if a reference spectrum file is loaded, this information may not be used anymore
            self.refspec = spectralinvariants.referencealbedo_transformed()
            self.refspecname = "Transformed_PROSPECT"
            self.wl_spec = spectralinvariants.reference_wavelengths()
            self.listbox.delete( 0, END )
            self.listbox.insert( END , self.refspecname )
            self.use_refspec_file = False # whether to load reference spectra from file
            
    def buttonplotref( self ):
        """
        plot a reference spectrum
        this function clears the plot as, potentially, the selection has changed
        """
        # load the spectra if needed
        if self.use_refspec_file:
            leafspectra = np.loadtxt(self.filename1)
            self.refspec = leafspectra[:, self.listbox.curselection()[0]+1 ] # 1st column is wavelength, hence +1
            self.refspecname = self.refspectranames[ self.listbox.curselection()[0]+1  ]
        
        if self.fig_refspec == None:
            self.fig_refspec = plt.figure()
        self.fig_refspec.clf()
        self.ax_refspec = self.fig_refspec.add_subplot(1, 1, 1) # handle for the axes in refspec
        self.ax_refspec.plot( self.wl_spec, self.refspec,'r-')
        self.ax_refspec.set_xlabel( 'Wavelength (nm)' ) # XXX, the units should still be checked
        self.ax_refspec.set_ylabel( 'reference spectrum: ' + self.refspecname )
        self.fig_refspec.canvas.draw()
        self.fig_refspec.show() 

    def buttondatafile( self ):
        """
        get data file name and load metadata
        """
        # reset openfilelist and other environment as if no data file were loaded
        self.hypdata_loaded = False
        self.openfilelist = []
#        self.button_datafile.configure( background='SystemButtonFace' )
        self.button_datafile.configure()
        self.button_run.configure( state=DISABLED )
        self.button_p.configure( state=DISABLED )
        self.filename2 = ""
        
        self.filename2 =  filedialog.askopenfilename(initialdir = self.hypdatadir, title = "Hyperspectal data file", filetypes = (("ENVI header files","*.hdr"),("all files","*.*")))
        
        if self.filename2 != "" :
            # open the data file -- reads only metadata
            self.hypdata = spectral.open_image(self.filename2)
            # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
            # print(self.hypdata.metadata.keys())
            if 'wavelength' in self.hypdata.metadata.keys():
                # wavelengths should be in metadata
                # these will be stored in the class for other functions to use (interpolation and plotting of reference data)
                self.wl_hyp = np.array(self.hypdata.metadata['wavelength'],dtype='float')
                if self.wl_hyp.max() < 100:
                    # in microns, convert to nm
                    self.wl_hyp *= 1000
                
                if self.hypdata.interleave == 1:
                    print("Band interleaved (BIL)")
                else:
                    print( self.filename2 + " not BIL -- opening still as BIL -- will be slower" )
                self.hypdata_map = self.hypdata.open_memmap()
                print("dimensions ", self.hypdata_map.shape) #shape[0]==lines, shape[1]==pixels, shape[2]==bands
                
                # save the handles to the 0th element of openfilelist
                self.openfilelist = [ [ self.filename2 , self.hypdata, self.hypdata_map ] ]
            
                self.plothypdata()
                
                # clear listbox and load the spectral bands in the hyperspectral data file
                self.listbox_wl.delete( 0, END )
                for item in self.wl_hyp:
                    self.listbox_wl.insert( END , str(item) )
                
                # set the initial selection    
                listboxselection_wl = np.where( np.logical_and(self.wl_hyp > 709,self.wl_hyp<791) )[0]
                self.listbox_wl.see( listboxselection_wl[0] ) # asks index to be visible
                # self.listbox_wl.index( listboxselection_wl[0] ) # asks index to be at top
                self.listbox_wl.selection_set( listboxselection_wl[0], listboxselection_wl[-1] )           
        
                self.hypdata_loaded = True
                self.button_datafile.configure(bg="green")

                self.button_run.configure( state=ACTIVE )
                self.button_p.configure( state=ACTIVE )
                self.plotrefspectrum2()
            else:
                print("Cannot load "+self.filename2)
                print("No wavelength information in file.")
                
                
                
    def plothypdata( self ):
        """
        function to (re)plot hyperspectral data
        assumes existence of
            self.openfilelist[]
            self.fig_hypdata
        """

        self.hypdata = self.openfilelist[0][1]
        self.hypdata_map = self.openfilelist[0][2]

        # choose bands for plotting
        i_r =  abs(self.wl_hyp-680).argmin() # red band
        i_g =  abs(self.wl_hyp-550).argmin() # green 
        i_b =  abs(self.wl_hyp-450).argmin() # blue
        
        # plot using pyplot.imshow -- this allows to catch clicks in the window
        hypdata_rgb = self.hypdata.read_bands([i_r,i_g,i_b]).astype('float32') 
        # datascaling = hypdata_rgb.max()
        # datascaling /= 0.95 
        ii = hypdata_rgb > 0
        datascaling = np.percentile( hypdata_rgb[ii], 98 )
        hypdata_rgb /= datascaling
        # percentile alone seems to give not so nice plots
        hypdata_rgb[ hypdata_rgb>1 ] = 1
        if self.fig_hypdata != None:
            # just in case, delete the previous figure and hopefully erase all taces of it
            print( self.fig_hypdata.number, plt.get_fignums() )
            plt.close( self.fig_hypdata )
        self.fig_hypdata = plt.figure() # create a new figure
        ax0 = self.fig_hypdata.add_subplot(1,1,1)
        ax0.imshow( hypdata_rgb ) # ax0 is self.fig_hypdata.axes[0]
        plottitle = os.path.split( self.openfilelist[0][0] )[1]
        plottitle = plottitle.split('.')[0]
        ax0.set_title(plottitle)
        
        self.fig_hypdata.canvas.draw()
        self.fig_hypdata.show()
        
        self.button_clearpoints.configure( state=DISABLED ) # fresh window: no points to clear

    def plotrefspectrum2( self ):
        """
        replot reference spectrum plot and add new data to reference spectrum plot
        plot the reference spectrum at wavelengths of hyperspectral data (with different color)
        """
        # first, replot as selection may have changed
        self.buttonplotref()
        
        # load the reference spectrum if needed
        if self.use_refspec_file:
            leafspectra = np.loadtxt(self.filename1)
            self.refspec = leafspectra[:, self.listbox.curselection()[0]+1 ] # 1st column is wavelength, hence +1

        # interpolate reference spectrum to hyperspectral wavelengths
        refspec_hyp = np.interp( self.wl_hyp, self.wl_spec, self.refspec )
        self.ax_refspec.plot( self.wl_hyp, refspec_hyp, 'b-' )
        # create a subset of reference data (interpolated to hyperspectral bands)
        refspec_hyp_subset = refspec_hyp[ np.array(self.listbox_wl.curselection()) ] # the reference spectrum subset to be used in calculations of p and DASF 
        # ??? tuples cannot be used to subset np.arrays?
        # plot the actual used hyperspectral bands as symbols)
        self.ax_refspec.plot( self.wl_hyp[ np.array(self.listbox_wl.curselection()) ], refspec_hyp_subset , 'go' )
        
    def clearpoints( self ):
        """
        Clear previous clicks in hypdata window
        """
        while len( self.fig_hypdata.axes[0].get_lines() ) > 0:
            self.fig_hypdata.axes[0].get_lines()[0].remove()
        self.fig_hypdata.canvas.draw()
        self.button_clearpoints.configure( state=DISABLED )
        
    def p_pixel( self, event=None ):
        """
        Calculate p, DASF and intercept, plot the spectrum and p-plot for a pixel.
        Can be run to initiate and cancel pixel picking, and also to calculate p.
        """
        if self.button_p.cget('text') == 'p for pixel':
            # initiate click collection
            if not self.hypdata_ciglock:
                if self.fig_hypdata.number not in plt.get_fignums():
                    print("p_pixel(): hyperspectral data window closed. Trying to re-create")
                    self.plothypdata()
                self.hypdata_ciglock = True # lock the cig for fig_hypdata so no other function can interfere
                self.button_p.configure( text='CLICK IN IMAGE', background='red' )
                self.catch_cid = self.fig_hypdata.canvas.mpl_connect('button_press_event',self.p_pixel)
            else:
                print("p_pixel(): hypdata_ciglock set to true, cannot access the figure.")
        elif event == None:
            # button was clicked, but we were waiting for a click in figure fig_hypdata
            # probably, the user wants to cancel
            print("p_pixel: canceling pixel selection")
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1
            self.button_p.configure( text='p for pixel', background='SystemButtonFace' )
            self.hypdata_ciglock = False # release lock on cig for fig_hypdata 
        
        else: # assume click was caught in the plot window
            print("p_pixel(): clicked "+str(event.xdata)+','+str(event.ydata))
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1
            
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            
            reflectance = self.hypdata_map[y,x,:]
            if int(self.hypdata.metadata['data type']) in (1,2,3,12,13):
                # these are integer data codes. assume it's reflectance*10,000
                reflectance = ( self.hypdata_map[y,x,:].astype('float32') )/10000

            # plot spectrum
            if self.fig_spectrum == None:
                self.fig_spectrum = plt.figure()
            self.fig_spectrum.clf()
            self.ax_spectrum = self.fig_spectrum.add_subplot(1, 1, 1) # handle for the axes in fig_spec
            self.ax_spectrum.plot( self.wl_hyp, self.hypdata_map[y,x,:], 'r-' )
            self.ax_spectrum.set_xlabel( 'Wavelength (nm)' )
            self.ax_spectrum.set_ylabel( 'Value' )
            self.ax_spectrum.set_title(str(x)+','+str(y))   
            self.fig_spectrum.canvas.draw()
            self.fig_spectrum.show()                     
            # mark the spot
            self.fig_hypdata.axes[0].plot( x, y, 'rx' )
            self.button_clearpoints.configure( state=ACTIVE )
            # display the figure
            self.fig_hypdata.canvas.draw()
            # run p calculations
            
            listboxselection_wl = self.listbox_wl.curselection()
            if len(listboxselection_wl) < 2:
                print("p_pixel(): too few wavelengths selected: ", len(listboxselection) )
            else: # all set!
                # read reference spetra (if needed) and interpolate to hyperspectral bands 
                if self.use_refspec_file:
                    leafspectra = np.loadtxt(self.filename1)        
                    # first column in leafspectra is wavelengths
                    self.wl_spec = leafspectra[:,0]
                    # which spectra should we use?
                    refspecno = self.listbox.curselection()
                    self.refspec = leafspectra[ :, refspecno[0]+1 ] # first column is wl, hence the "+1" 
                    
                # np.interp does not check that the x-coordinate sequence xp is increasing. If xp is not increasing, the results are nonsense. A simple check for increasing is:
                refspec_hyp = np.interp( self.wl_hyp, self.wl_spec, self.refspec )
                
                # subset of reference data (interpolated to hyperspectral bands)
                ii = np.array(listboxselection_wl) 
                refspec_hyp_subset = refspec_hyp[ ii ] # the reference spectrum subset to be used in calculations of p and DASF
                BRF_subset = reflectance[ii] # convert to reflectance units

                pvec = p( BRF_subset, refspec_hyp_subset )
                # p_values:output, ndarray of length 4
                # 0:slope 1:intercept 2: DASF 3:R
                
                #plot
                if self.fig_pplot == None:
                    self.fig_pplot = plt.figure()
                self.fig_pplot.clf()
                self.ax_pplot = self.fig_pplot.add_subplot(1,1,1)
                self.ax_pplot.plot( BRF_subset, BRF_subset/refspec_hyp_subset, 'ro' )
                
                gx = np.linspace( BRF_subset.min(), BRF_subset.max(), 3 )
                gy = pvec[0]*gx + pvec[1]
                self.ax_pplot.plot( gx, gy, 'g-')
                self.ax_pplot.set_xlabel( "BRF" )
                self.ax_pplot.set_ylabel( "BRF/w" )
                self.ax_pplot.set_title(str(x)+','+str(y))
                self.fig_pplot.canvas.draw()
                self.fig_pplot.show()
                
                print( "p=%6.3f, intercept=%6.3f, DASF=%6.3f, R=%5.2f" % tuple(pvec) )
                        
            # finish and reset
            self.button_p.configure( text='p for pixel', background='SystemButtonFace' )
            self.hypdata_ciglock = False # release lock on cig for fig_hypdata 
 

    def buttonrun( self ):
        """
        Function to run the computations on the whole image or to stop the running omputations,
        depending on the state wich is determined from the text on the button
        """
           
        if self.button_run.cget('text')=='Stop':
            # the thread should be running, stop it
            self.progressvar_p.set(-1) # this signals break
            self.button_quit.configure( state=ACTIVE )
            self.button_datafile.configure( state=ACTIVE )
            self.button_spectrumfile.configure( state=ACTIVE )
            self.button_run.configure( text='Run', state=DISABLED ) # the button will be enabled once the thread exits
            print("Break signal caught")
        else:
            # sanity check: do we have selections in both listboxes
            allset = True
            if self.use_refspec_file:
                listboxselection = self.listbox.curselection()
                refspec = None
                refspecname = None
                if listboxselection == ():
                    print("No reference spectrum selected")
                    allset = False
            else:
                listboxselection = None # self.refspec will be used instead
                refspec = self.refspec
                refspecname = self.refspecname
            listboxselection_wl = self.listbox_wl.curselection()
            if len(listboxselection_wl) < 2:
                print("Too few wavelengths selected: ", len(listboxselection) )
                allset = False
            # where to save the new p-data:
            filename3 =  filedialog.asksaveasfilename(initialdir = self.hypdatadir, title = "p-file name to create",filetypes = (("ENVI hdr files","*.hdr"),("all files","*.*")))
            if filename3 == '':
                allset = False
                
            if allset:
                # where to save the new p-data:
                if filename3[-4:] != ".hdr":
                    filename3 += ".hdr"
                # get the handles of the hyperspectral data
                if self.openfilelist[0][1] != None:
                    # the file has been opened, this should always be the case
                    self.filename2 = self.openfilelist[0][0]
                    self.hypdata = self.openfilelist[0][1]
                    self.hypdata_map = self.openfilelist[0][2]
                else:
                    # this should actually never happen, but still...
                    print("Warning: filename2 has not been opened. Strange, this should never happen.")
                    # the file will be opened in p_processing()
                    self.hypdata = None
                    self.hypdata_map = None
                
                # create thread
                self.thread1 = p_thread( self.master, self.filename1, listboxselection[0], listboxselection_wl, self.filename2, filename3, self.hypdata, self.hypdata_map, 
                    self.progressvar_p, refspec, refspecname )
                
                # prepare GUI and start thread
                # disable all unnecessary stuff here
                self.button_quit.configure( state=DISABLED )
                self.button_datafile.configure( state=DISABLED )
                self.button_spectrumfile.configure( state=DISABLED )
                self.button_p.configure( state=DISABLED )
                self.button_run.configure( text='Stop' )
                self.progressvar_p.set(0) 
                self.thread1.start()
                
            else:
                print("Cannot run.")
        
if __name__ == '__main__':
    root = Tk()
    my_gui = pGUI(root)
    root.mainloop()
