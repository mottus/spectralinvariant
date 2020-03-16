from tkinter import *
from tkinter import filedialog, ttk
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import threading
import time
import os

from spectralinvariant.hypdatatools_img import *
from spectralinvariant.hypdatatools_algorithms import *

# script to calculate the spectral scattering coefficient W.
# inputs: hyperspectral image (i.e., BRF) and a file with a layer named DASF
# according to the spectral invariant theory, BRF = W*DASF


class W_thread( threading.Thread ):
    """
    the thread which will run the process
    """
    def __init__( self, tkroot, filename1, filename2, filename3, DASFnumber, hypdata=None, hypdata_map=None, DASFdata=None, DASFdata_map=None, progressvar=None ):
        """
        the inputs are
        tkroot: handle for the root window for signaling
        the rest are forwarded to p_processing()
        """
        threading.Thread.__init__(self) # this call is apparently mention as a requirement in some tutorial
        self.tkroot = tkroot
        self.filename1 = filename1
        self.filename2 = filename2
        self.filename3 = filename3
        self.DASFnumber = DASFnumber
        self.hypdata = hypdata
        self.hypdata_map = hypdata_map
        self.DASFdata = DASFdata
        self.DASFdata_map = DASFdata_map    
        self.progressvar = progressvar
    def run(self):
        """
        a wrapper for running p_processing in a separate thread
        """
        # do the thing
        W_processing( self.filename1, self.filename2, self.filename3, self.DASFnumber, self.hypdata, self.hypdata_map, self.DASFdata, self.DASFdata_map, progressvar=self.progressvar )
        # signal that we have finished
        print("W processing thread exits.")
        self.tkroot.event_generate("<<thread_end>>", when="tail")
        
class WGUI:
    """
    the GUI for creating the thread which will do the calulation
    """    
    openfilelist = [] # list of loaded hyperspectral file names and handles
        # each element is a list [ filename filehandle datahandle ]
        # intially, when loading the hyperspectral handles are set to None; they are assigned whn file is opened for e.g. plotting
        # the list will contain only one element as only one file is opened in this program
    dasffilelist = [] # same as openfilelist, but for DASF file + element [3] = list of band names
    
    hypdatadir="D:\\mmattim\\wrk\\hyytiala-D\\" # location of hyperspectral data, initial suggestion in file dialog
    
    DASF_loaded = False # flag
    hypdata_loaded = False # flag

    
    def __init__(self, master):
        self.master = master
        master.title("GUI for calculating W")
        self.progressvar_W = DoubleVar() # variable to signal progress and breaks
        self.progressvar_W.set(0) # just in case
        # initialize DASF optiomenu to empty
        self.DASFband = StringVar(master) # tkinter string to set and get the value in DASF optionmenu
        self.DASFband.set("not loaded")

        # a frame for buttons to load stuff etc.
        bw = 25 # button width
        self.frame_buttons = Frame(master) #just in case, put everythin in a frame, stuff can be added later if needed
        self.button_quit = Button( self.frame_buttons, text='Quit', width=bw, command=self.buttonquit )
        self.button_datafile = Button( self.frame_buttons, text='Load datafile', width=bw, command=self.datafile )
        self.button_DASFfile = Button( self.frame_buttons, text='Load DASF file', width=bw, command=self.DASFfile )
        self.option_DASFband = OptionMenu( self.frame_buttons, self.DASFband, "" )
        # for some strange reason, I could not configure OptionMenu as any other widget
        self.option_DASFband['width'] = bw-5
        self.option_DASFband['state'] = DISABLED
        self.button_plotDASF = Button( self.frame_buttons, text='Plot DASF', width=bw, command=self.plotDASF, state=DISABLED )
        self.button_run = Button( self.frame_buttons, text='Calculate W for image', width=bw, command=self.buttonrun, state=DISABLED )
        self.progressbar = ttk.Progressbar( self.frame_buttons, orient='horizontal', maximum=1, value=0, variable=self.progressvar_W, mode='determinate' )
        self.button_datafile.pack( side='top' )
        self.button_DASFfile.pack( side='top' )
        self.option_DASFband.pack( side='top' )
        self.button_plotDASF.pack( side='top' )
        self.button_run.pack( side='top' )
        self.progressbar.pack( fill=X, side='bottom' )
        self.button_quit.pack( side='bottom' )

        self.frame_buttons.pack( side='left' )
        
        master.bind("<<thread_end>>", self.thread_end )

        
    def datafile( self ):
        """
        get data file name and load metadata
        """
        # reset openfilelist and other environment as if no data file were loaded
        self.hypdata_loaded = False
        self.openfilelist = []
        self.button_datafile.configure( background='SystemButtonFace' )
        self.button_run.configure( state=DISABLED )
        
        filename1 =  filedialog.askopenfilename(initialdir = self.hypdatadir, title = "Choose hyperspectal data file", filetypes = (("ENVI header files","*.hdr"),("all files","*.*")))
        
        if filename1 != "" : 
        
            # open the data file -- reads only metadata
            hypdata = spectral.open_image(filename1)
            # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
            # print(hypdata.metadata.keys())
            if 'wavelength' in hypdata.metadata.keys():
                if hypdata.interleave == 1:
                    print("Band interleaved (BIL)")
                else:
                    print( filename1 + " not BIL -- opening still as BIL -- will be slower" )
                hypdata_map = hypdata.open_memmap()
                print("dimensions ",hypdata_map.shape) #shape[0]==lines, shape[1]==pixels, shape[2]==bands
                
                # save the handles to the 0th element of openfilelist
                self.openfilelist = [ [ filename1 , hypdata, hypdata_map ] ]
                
                self.fig_hypdata = plot_hyperspectral( filename1, hypdata, hypdata_map )
                
                self.wl_hyp, wl_found = get_wavelength( filename1, hypdata )
                
                self.hypdata_loaded = True
                self.button_datafile.configure( bg='green' )
                if self.DASF_loaded:
                    self.button_run.configure( state=ACTIVE )
            else:
                print("No spectral information, cannot load file "+filename1)
            
    def DASFfile(self):
        """
        Choose the file containing DASF, save it in self.filename2 and plot
        """
        # reset DASFfilelist and other environment as if no data file were loaded
        self.DASF_loaded = False
        self.DASFfilelist = []
        self.button_DASFfile.configure( background='SystemButtonFace' )
        self.button_run.configure( state=DISABLED )
        self.option_DASFband['state'] = DISABLED
        self.button_plotDASF.configure( state=DISABLED )
            
        filename2 =  filedialog.askopenfilename(initialdir = self.hypdatadir, title = "Choose DASF file", filetypes = (("ENVI header files","*.hdr"),("all files","*.*")))
        
        if filename2 != "" :
            # open the data file -- reads only metadata
            DASFdata = spectral.open_image(filename2)
            # DASFdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
            # print(DASFdata.metadata.keys())
            if DASFdata.interleave == 1:
                print("Band interleaved (BIL)")
            else:
                print( filename2 + " not BIL -- opening still as BIL -- will be slower" )
            DASFdata_map = DASFdata.open_memmap()
            print("dimensions ",DASFdata_map.shape) #shape[0]==lines, shape[1]==pixels, shape[2]==bands
            
            
            # find the DASF layer
            if 'band names' in DASFdata.metadata:
                DASFfilebands = DASFdata.metadata['band names']
                if 'DASF' in DASFdata.metadata['band names']:
                    self.DASFband.set('DASF')
                    print("Found DASF layer in "+filename2)
                else:
                    #no DASF in band names, let the user decide which one to use, set default to first
                    self.DASFband.set( DASFdata.metadata['band names'][0] )
            else:
                # no band name info given, use numbers and set default to first
                DASFfilebands = [str(i) for i in range( 1,DASFdata_map.shape[2]+1 ) ]
                self.DASFband.set( DASFfilebands[0] )
                
            # save the handles to the 0th element of DASFfilelist
            self.DASFfilelist = [ [ filename2 , DASFdata, DASFdata_map, DASFfilebands ] ]

            # fill the DASF optionmenu
            self.option_DASFband['menu'].delete( 0, END )
            for choice in DASFfilebands:
                self.option_DASFband['menu'].add_command(label=choice, command=lambda v=choice: self.DASFband.set(v))
            
            self.option_DASFband['state'] = ACTIVE
            self.button_plotDASF.configure( state=ACTIVE )
            self.DASF_loaded = True
            self.button_DASFfile.configure( background='green' )
            if self.hypdata_loaded:
                self.button_run.configure( state=ACTIVE )
                
    def plotDASF(self):
        """ 
        plot the DASF using imshow
        """
        DASFnumber = self.DASFfilelist[0][3].index(self.DASFband.get())
        print( "Plotting DASF layer " + self.DASFband.get() +" [" + str(DASFnumber)+"]" )
        
        DASF_number = [ DASFnumber ]
        DASF_filename = self.DASFfilelist[0][0]
        DASF_data = self.DASFfilelist[0][1]
        DASF_memmap = self.DASFfilelist[0][2]
        
        DASF_figurehandle = plot_singleband( DASF_filename, DASF_data, DASF_memmap, DASF_number )
        
    def thread_end(self,*args):
        """
        function to restore stuff for potential next processing
        """
        self.button_quit.configure( state=ACTIVE )
        self.button_datafile.configure( state=ACTIVE )
        self.button_DASFfile.configure( state=ACTIVE )
        self.button_run.configure( state=ACTIVE )
        print("Thread exit caught")

    def buttonquit(self):
        """
        function to end the misery
        note: the pyplot windows are not closed. Maybe, it would be nice to keep track of those to close them
        """
        self.master.destroy() # destruction of root required for program to continue (to its end)
        
    def buttonrun(self):
        """
        Function to run the computations on the whole image or to stop the running omputations,
        depending on the state wich is determined from the text on the button
        """
           
        if self.button_run.cget('text')=='PROCESSING. Click to stop':
            # the thread should be running, stop it
            self.progressvar_W.set(-1) # this signals break
            self.button_quit.configure( state=ACTIVE )
            self.button_datafile.configure( state=ACTIVE )
            self.button_DASFfile.configure( state=ACTIVE )
            self.button_run.configure( text='Calculate W for image', state=DISABLED ) # the button will be enabled once the thread exits
            print("Break signaled")
        else:                 
            # sanity check: do we have selections in both listboxes
            allset = True
            # placeholder for basic sanity checks (which are not implemented)
            
            
            # where to save the new p-data:
            filename3 =  filedialog.asksaveasfilename(initialdir = self.hypdatadir, title = "W file to create",filetypes = (("ENVI hdr files","*.hdr"),("all files","*.*")))
            if filename3 == '':
                allset = False
            
            if allset:
                # where to save the new p-data:
                if filename3[-4:] != ".hdr":
                    filename3 += ".hdr"
                filename1 = self.openfilelist[0][0]
                # get the handles of the hyperspectral data
                if self.openfilelist[0][1] != None:
                    # the file has been opened, this should always be the case
                    hypdata = self.openfilelist[0][1]
                    hypdata_map = self.openfilelist[0][2]
                else:
                    # this should actually never happen, but still...
                    print("Warning: hyperspectral file has not been opened. Strange, this should never happen.")
                    # the file will be opened in p_processing()
                    
                filename2 = self.DASFfilelist[0][0]
                # get the handles of the DASF data
                if self.DASFfilelist[0][1] != None:
                    # the file has been opened, this should always be the case
                    DASFdata = self.DASFfilelist[0][1]
                    DASFdata_map = self.DASFfilelist[0][2]
                    DASFnumber = self.DASFfilelist[0][3].index( self.DASFband.get() )
                else:
                    # this should actually never happen, but still...
                    print("Warning: DASF file has not been opened. Strange, this should never happen.")
                    # the file will be opened in p_processing(), band #0 will be used
                    DASFnumber = 0
    
                # create thread
                self.thread1 = W_thread( self.master, filename1, filename2, filename3, DASFnumber, hypdata, hypdata_map, DASFdata, DASFdata_map, self.progressvar_W )
                
                # prepare GUI and start thread
                # disable all unnecessary stuff here
                self.button_quit.configure( state=DISABLED )
                self.button_datafile.configure( state=DISABLED )
                self.button_DASFfile.configure( state=DISABLED )
                self.button_run.configure( text='PROCESSING. Click to stop' )
                self.progressvar_W.set(0) 
                self.thread1.start()
            else:
                print("Things not set up properly, not starting processing")

        
if __name__ == '__main__':
    root = Tk()
    GUI = WGUI( root )
    root.mainloop()
    

    