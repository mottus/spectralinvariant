from tkinter import *
from tkinter import filedialog, ttk
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import threading
import os

from spectralinvariant.hypdatatools_algorithms import PRI_processing, PRI_singlepoint

# script to calculate the dependence of PRI on shadow fraction
# inputs: hyperspectral image (i.e., BRF) and a file with a layer 'intercept'
# according to the spectral invariant theory, sunlit fraction ~ intercept (denoted as rho)
# NOTE: the actual output is dependence of PRI on rho which can be converted to sunlit fraction using knowledge on G and solar angle
# to get the dependence, an area is always sampled. Check the code for the specific shape and size of the area
    
    
class PRI_thread( threading.Thread ):
    """
    the thread which will run the process without freezing GUI
    """
    def __init__( self, tkroot, filename1, filename2, filename3, rhonumber, windowsize, hypdata=None, hypdata_map=None, rhodata=None, rhodata_map=None, progressvar=None ):
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
        self.windowsize = windowsize
        self.rhonumber = rhonumber
        self.hypdata = hypdata
        self.hypdata_map = hypdata_map
        self.rhodata = rhodata
        self.rhodata_map = rhodata_map    
        self.progressvar = progressvar
    def run(self):
        """
        a wrapper for running p_processing in a separate thread
        """
        # do the thing
        PRI_processing( self.filename1, self.filename2, self.filename3, self.rhonumber, self.windowsize, self.tkroot, self.hypdata, self.hypdata_map, self.rhodata, self.rhodata_map, progressvar=self.progressvar )
        # signal that we have finished
        self.tkroot.event_generate("<<thread_end>>", when="tail")
        
class PRIGUI:
    """
    the GUI for creating the thread which will do the calculation
    """
    
    openfilelist = [] # list of loaded hyperspectral file names and handles
        # each element is a list [ filename filehandle datahandle ]
        # intially, when loading the hyperspectral handles are set to None; they are assigned whn file is opened for e.g. plotting
        # the list will contain only one element as only one file is opened in this program
    rhofilelist = [] # same as openfilelist, but for rho file + element [3] = list of band names
    
    hypdatadir="D:\\mmattim\\wrk\\hyytiala-D\\" # location of hyperspectral data, initial suggestion in file dialog
    
    rho_loaded = False # flag
    hypdata_loaded = False # flag
    catch_cid = -1 # connection id for pyplot mouse click
    fig_hypdata = None #image handle
    hypdata_ciglock = False # lock for catching clicks in fig_hypdata -- only one function can catch them at a time
    
    def __init__(self, master):
        self.master = master
        master.title("GUI for calculating PRI~sunlit fraction")
        self.progressvar_PRI = DoubleVar() # for signaling progress and breaks
        self.progressvar_PRI.set(0) # just in case
        # initialize rho optiomenu to empty
        self.rhoband = StringVar() # tkinter string to set and get the value in rho optionmenu
        self.rhoband.set("not loaded")
        self.PRIwindow = StringVar() # tkinter string to set and get the value in PRI window optionmenu
        PRIwindow_optionlist = ['window '+str(i)+'x'+str(i) for i in range(5,31,4) ]
        self.PRIwindow.set(PRIwindow_optionlist[-1])
        self.hypdata_ciglock = False 

        # a frame for buttons to load stuff etc.
        bw = 25 # button width
        self.frame_buttons = Frame(master) #just in case, put everythin in a frame, stuff can be added later if needed
        self.button_quit = Button( self.frame_buttons, text='Quit', width=bw, command=self.buttonquit )
        self.button_datafile = Button( self.frame_buttons, text='Load datafile', width=bw, command=self.datafile )
        self.button_rhofile = Button( self.frame_buttons, text='Load intercept file', width=bw, command=self.rhofile )
        self.option_rhoband = OptionMenu( self.frame_buttons, self.rhoband, "" )
        # for some strange reason, I could not configure OptionMenu as any other widget
        self.option_rhoband['width'] = bw-5
        self.option_rhoband['state'] = DISABLED
        self.button_plotrho = Button( self.frame_buttons, text='Plot intercept', width=bw, command=self.plotrho, state=DISABLED )
        self.button_plotspectrum = Button( self.frame_buttons, text='Plot pixel spectrum', width=bw, command=self.plotspectrum, state=DISABLED )
        self.option_PRIwindow = OptionMenu( self.frame_buttons, self.PRIwindow, *PRIwindow_optionlist )
        # for some strange reason, I could not configure OptionMenu as any other widget
        self.option_PRIwindow['width'] = bw-5
        self.button_plotpixel = Button( self.frame_buttons, text='Plot pixel PRI', width=bw, command=self.plotpixel, state=DISABLED )
        self.button_clearpoints = Button( self.frame_buttons, text='Clear points', width=bw, command=self.clearpoints, state=DISABLED )
        self.button_run = Button( self.frame_buttons, text='Run for whole image', width=bw, command=self.buttonrun, state=DISABLED )
        # self.button_stop = Button( self.frame_buttons, text='Stop', width=bw, command=self.buttonstop, state=DISABLED )
        self.progressbar = ttk.Progressbar( self.frame_buttons, orient='horizontal', maximum=1, value=0, variable=self.progressvar_PRI, mode='determinate' )
        self.button_datafile.pack( side='top' )
        self.button_rhofile.pack( side='top' )
        self.option_rhoband.pack( side='top' )
        self.button_plotrho.pack( side='top' )
        self.button_plotspectrum.pack( side='top' )
        self.option_PRIwindow.pack( side='top' )
        self.button_plotpixel.pack( side='top' )
        self.button_clearpoints.pack( side='top' )
        self.button_run.pack( side='top' )
        # self.button_stop.pack( side='top' )
        self.progressbar.pack( fill=X, side='bottom' )
        self.button_quit.pack( side='bottom' )

        self.frame_buttons.pack( side='left' )
        
        # signal to learn that processing has finished
        master.bind( "<<thread_end>>", self.thread_end )
        
    def datafile( self ):
        """
        get data file name and load metadata
        """
        # reset openfilelist and other environment as if no data file were loaded
        self.hypdata_loaded = False
        self.openfilelist = []
        self.button_datafile.configure( background='SystemButtonFace' )
        self.button_plotspectrum.configure( text='Plot pixel spectrum', background='SystemButtonFace', state=DISABLED )
        self.button_plotpixel.configure( text='Plot pixel PRI', background='SystemButtonFace', state=DISABLED )
        self.button_run.configure( state=DISABLED )
        # reset also possible click catch attempts in the figure (which is redrawn)
        self.hypdata_ciglock = False # release lock on cig for fig_hypdata
        if self.catch_cid != -1:
            # there seems to be a connection with mouse clicks, release it
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1

        filename1 =  filedialog.askopenfilename(initialdir = self.hypdatadir, title = "Choose hyperspectal data file", filetypes = (("ENVI header files","*.hdr"),("all files","*.*")))
        
        if filename1 != "" : 
        
            # open the data file -- reads only metadata
            hypdata = spectral.open_image(filename1)
            # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
            # print(hypdata.metadata.keys())
            if hypdata.interleave == 1:
                print("Band interleaved (BIL)")
            else:
                print( filename1 + " not BIL -- opening still as BIL -- will be slower" )
            hypdata_map = hypdata.open_memmap()
            print("dimensions ",hypdata_map.shape) #shape[0]==lines, shape[1]==pixels, shape[2]==bands
            
            # save the handles to the 0th element of openfilelist
            self.openfilelist = [ [ filename1 , hypdata, hypdata_map ] ]
            
            # wavelengths should be in metadata
            # these will be stored in the class for other functions to use (interpolation and plotting of reference data)
            if 'wavelength' in hypdata.metadata.keys():
                self.wl_hyp = np.array(hypdata.metadata['wavelength'],dtype='float')
                if self.wl_hyp.max() < 100:
                    # in microns, convert to nm
                    self.wl_hyp *= 1000
                
                self.plothypdata()
                
                self.hypdata_loaded = True
                self.button_datafile.configure( bg='green' )
                self.button_plotspectrum.configure( state=ACTIVE )
                if self.rho_loaded:
                    self.button_run.configure( state=ACTIVE )
                    self.button_plotpixel.configure( state = ACTIVE )
            else: # if 'wavelength' in hypdata.metadata.keys():
                print("No spectral information available in file "+filename1)
                
    def plothypdata( self ):
        """
        function to (re)plot hyperspectral data
        assumes existence of
            self.openfilelist[]
            self.fig_hypdata
        """

        hypdata = self.openfilelist[0][1]
        hypdata_map = self.openfilelist[0][2]
        
        wl_hyp = np.array(hypdata.metadata['wavelength'],dtype='float')
        if wl_hyp.max() < 100:
            # in microns, convert to nm
            wl_hyp *= 1000

        # choose bands for plotting
        i_r =  abs(wl_hyp-680).argmin() # red band
        i_g =  abs(wl_hyp-550).argmin() # green 
        i_b =  abs(wl_hyp-450).argmin() # blue
        
        # set max area
        # plotsize_max = 1024
        # plot_pixmax = min(plotsize_max,hypdata_map.shape[1])
        # plot_rowmax = min(plotsize_max,hypdata_map.shape[0])
        # plot using spectral
        # self.fig_hypdata = spectral.imshow(hypdata[0:plot_rowmax,0:plot_pixmax,:], (i_r,i_g,i_b)  )
        
        # plot using pyplot.imshow -- this allows to catch clicks in the window
        hypdata_rgb = hypdata.read_bands([i_r,i_g,i_b]).astype('float32') 
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


    def rhofile(self):
        """
        Choose the file containing intercept (rho), save it in self.filename2 and plot
        """
        # reset rhofilelist and other environment as if no data file were loaded
        self.rho_loaded = False
        self.rhofilelist = []
        self.button_rhofile.configure( background='SystemButtonFace' )
        self.button_run.configure( state=DISABLED )
        self.option_rhoband['state'] = DISABLED
        self.button_plotrho.configure( state=DISABLED )
            
        filename2 =  filedialog.askopenfilename(initialdir = self.hypdatadir, title = "Choose intercept file", filetypes = (("ENVI header files","*.hdr"),("all files","*.*")))
        
        if filename2 != "" :
            # open the data file -- reads only metadata
            rhodata = spectral.open_image(filename2)
            # rhodata.metadata is of type dict, use e.g. hypdata.metadata.keys()
            # print(rhodata.metadata.keys())
            if rhodata.interleave == 1:
                print("Band interleaved (BIL)")
            else:
                print( filename2 + " not BIL -- opening still as BIL -- will be slower" )
            rhodata_map = rhodata.open_memmap()
            print("dimensions ",rhodata_map.shape) #shape[0]==lines, shape[1]==pixels, shape[2]==bands
            
            
            # find the intercept (rho) layer
            if 'band names' in rhodata.metadata:
                rhofilebands = rhodata.metadata['band names']
                if 'intercept' in rhodata.metadata['band names']:
                    self.rhoband.set('intercept')
                    print("Found intercept layer in "+filename2)
                else:
                    #no intercept in band names, let the user decide which one to use, set default to first
                    self.rhoband.set( rhodata.metadata['band names'][0] )
            else:
                # no band name info given, use numbers and set default to first
                rhofilebands = [str(i) for i in range( 1,rhodata_map.shape[2]+1 ) ]
                self.rhoband.set( rhofilebands[0] )
                
            # save the handles to the 0th element of rhofilelist
            self.rhofilelist = [ [ filename2 , rhodata, rhodata_map, rhofilebands ] ]

            # fill the intercept (rho) optionmenu
            self.option_rhoband['menu'].delete( 0, END )
            for choice in rhofilebands:
                self.option_rhoband['menu'].add_command(label=choice, command=lambda v=choice: self.rhoband.set(v))
            
            self.option_rhoband['state'] = ACTIVE
            self.button_plotrho.configure( state=ACTIVE )
            self.rho_loaded = True
            self.button_rhofile.configure( background='green' )
            
            if self.hypdata_loaded:
                self.button_run.configure( state=ACTIVE )
                self.button_plotpixel.configure( state=ACTIVE )
                
    def plotrho(self):
        """ 
        plot the intercept (rho) using imshow
        """
        rhonumber = self.rhofilelist[0][3].index(self.rhoband.get())
        print( "Plotting intercept (rho) layer " + self.rhoband.get() +" [" + str(rhonumber)+"]" )
        
        rhonumber = [ rhonumber ]
        rho_memmap = self.rhofilelist[0][2]
        spectral.imshow( rho_memmap, rhonumber, stretch=0.98 )
        
    def plotspectrum( self, event=None ):
        """
        catch clicks in figure self.fig_hypdata and plot the spectrum
        action depends on the state of the button self.button_plotspectrum
        """
        if self.button_plotspectrum.cget('text') == 'Plot pixel spectrum':
            # initiate pixel spectrum collection, set up connection with figure window
            if not self.hypdata_ciglock: # only do sth if no other function is waiting for a click in fig_hypdata
                if self.fig_hypdata.number not in plt.get_fignums():
                    print("plotspectrum(): hyperspectral data window closed. Trying to re-create")
                    self.plothypdata()
                self.hypdata_ciglock = True # lock the cig for fig_hypdata so no other function can interfere
                self.button_plotspectrum.configure( text='CLICK IN IMAGE', background='red' )
                self.catch_cid = self.fig_hypdata.canvas.mpl_connect('button_press_event',self.plotspectrum)
            else:
                print("plotspectrum(): hypdata_ciglock set to true, cannot access the figure.")
        elif event==None:
            # button text was 'CLICK IN IMAGE', but the button itself was clicked.
            # probably, the user wants to cancel
            print("plotspectrum(): canceling pixel selection")
            self.button_plotspectrum.configure( text='Plot pixel spectrum', background='SystemButtonFace')
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1
            self.hypdata_ciglock = False # release lock on cig for fig_hypdata  
        else: 
            # we are called by pyplot event, click in fig_hypdata
            print("plotspectrum(): clicked "+str(event.xdata)+','+str(event.ydata))
            self.button_plotspectrum.configure( text='Plot pixel spectrum', background='SystemButtonFace' )
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1
            
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            
            # plot
            fig_spec = plt.figure() # handle for spectrum figures
            fig_spec.clf()
            ax_spec = fig_spec.add_subplot(1, 1, 1) # handle for the axes in fig_spec
            
            hypdata = self.openfilelist[0][1]
            hypdata_map = self.openfilelist[0][2]  
            wl = np.array(hypdata.metadata['wavelength'],dtype='float')
            if wl.max() < 100:
                # in microns, convert to nm
                wl *= 1000
            
            ax_spec.plot( wl, hypdata_map[y,x,:], 'r-' )
            ax_spec.set_xlabel( 'Wavelength' )
            ax_spec.set_ylabel( 'Value' )
            ax_spec.set_title(str(x)+','+str(y))
            fig_spec.show()
            
            i780 = ( (wl-780)**2 ).argmin()
            i680 = ( (wl-680)**2 ).argmin()
            NDVI = ( hypdata_map[y,x,i780] - hypdata_map[y,x,i680] ) / (hypdata_map[y,x,i780] + hypdata_map[y,x,i680] )
            print(str(x)+","+str(y)+" NDVI = " + str( round(NDVI,3) ) )
            
            # mark the spot
            self.fig_hypdata.axes[0].plot( x, y, 'bo', markerfacecolor='none' )
            self.fig_hypdata.canvas.draw()
            self.button_clearpoints.configure( state=ACTIVE )

            self.hypdata_ciglock = False # release lock on cig for fig_hypdata 
            
    def clearpoints( self ):
        """
        Clear previous clicks in hypdata window
        """
        while len( self.fig_hypdata.axes[0].get_lines() ) > 0:
            self.fig_hypdata.axes[0].get_lines()[0].remove()
        self.fig_hypdata.canvas.draw()
        self.button_clearpoints.configure( state=DISABLED )
        
    def plotpixel( self, event=None ):
        """
        catch clicks in figure self.fig_hypdata and plot the the PRI-rho relationship for a clicked pixel 
        action depends on the state of the button self.button_plotspectrum
        """
        if self.button_plotpixel.cget('text') == 'Plot pixel PRI':
            if not self.hypdata_ciglock: # only do sth if no other function is waiting for a click in fig_hypdata
                if self.fig_hypdata.number not in plt.get_fignums():
                    print("plotpixel(): hyperspectral data window closed. Trying to re-create")
                    self.plothypdata()
                self.hypdata_ciglock = True # lock the cig for fig_hypdata so no other function can interfere
                self.button_plotpixel.configure( text='CLICK IN IMAGE', background='red')
                self.catch_cid = self.fig_hypdata.canvas.mpl_connect('button_press_event',self.plotpixel)
            else:
                print("catchpixel(): hypdata_ciglock set to true, cannot access the figure.")
        elif event == None:
            # button text was 'CLICK IN IMAGE', but the button itself was clicked.
            # probably, the user wants to cancel
            print("plotpixel(): canceling pixel selection")
            self.button_plotpixel.configure( text='Plot pixel PRI', background='SystemButtonFace' )
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            self.catch_cid = -1
            self.hypdata_ciglock = False # release lock on cig for fig_hypdata 
        else:
            # we were called by a click in fig_hypdata, event has real data in it            
            self.button_plotpixel.configure( text='Plot pixel PRI', background='SystemButtonFace' )
            self.fig_hypdata.canvas.mpl_disconnect(self.catch_cid)
            # print("plotpixel(): clicked "+str(event.xdata)+','+str(event.ydata))
            # print(event.xdata,event.ydata)
            filename1 = self.openfilelist[0][0]
            hypdata = self.openfilelist[0][1]
            hypdata_map = self.openfilelist[0][2]            
            filename2 = self.rhofilelist[0][0]
            rhodata = self.rhofilelist[0][1]
            rhodata_map = self.rhofilelist[0][2]
            rhonumber = self.rhofilelist[0][3].index( self.rhoband.get() )
            N = int( self.PRIwindow.get().split('x')[-1] )
            
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            # mark the spot
            self.fig_hypdata.axes[0].plot( x, y, 'rx' )
            self.fig_hypdata.canvas.draw()
            self.button_clearpoints.configure( state=ACTIVE )

            
            PRI_singlepoint( filename1, filename2, x, y, N, rhonumber, hypdata, hypdata_map, rhodata, rhodata_map )
            self.hypdata_ciglock = False # release lock on cig for fig_hypdata 
        
        
    def thread_end(self,*args):
        """
        function to restore stuff for potential next processing
        """
        self.button_quit.configure( state=ACTIVE )
        self.button_datafile.configure( state=ACTIVE )
        self.button_rhofile.configure( state=ACTIVE )
        self.button_run.configure( text='Run for whole image')
        print("Thread exit caught")

    def buttonquit(self):
        """
        function to end the misery
        note: the pyplot windows are not closed. Maybe, it would be nice to keep track of those to close them
        """
        self.master.destroy() # destruction of root required for program to continue (to its end)
        
    def buttonstop( self ):
        """
        the old function to for the separate Stop button. Not used anymore, stop is integrated with buttonrun()
        """
        # the outputfile should be closed together with the namespace it's in
        self.progressvar_PRI.set(-1) # this will signal a break
        print("Break signaled")
        self.button_stop.configure( state=DISABLED )
        self.button_quit.configure( state=ACTIVE )
        self.button_datafile.configure( state=ACTIVE )
        self.button_rhofile.configure( state=ACTIVE )
        self.button_run.configure( state=ACTIVE )
        self.button_plotpixel.configure( state = ACTIVE )
        
    def buttonrun(self):
        """
        The function called when the Run/Stop button is clicked
        """
        
        #First check what is the text of the button now
        if self.button_run.cget('text') == 'Stop':
            # Abort processing
            print("Signaling break.")
            self.progressvar_PRI.set(-1) # this signals break
            self.button_quit.configure( state=ACTIVE )
            self.button_datafile.configure( state=ACTIVE )
            self.button_rhofile.configure( state=ACTIVE )
            self.button_run.configure( text='Run for whole image' )
            self.button_plotpixel.configure( state = ACTIVE )
        else:
            # The 'Run for whole image' button was clicked           
            
            # sanity check: do we have selections in both listboxes
            allset = True
            # placeholder for basic sanity checks (which are not implemented)
    
    
            # where to save the new p-data:
            filename3 =  filedialog.asksaveasfilename(initialdir = self.hypdatadir, title = "p-file name to create",filetypes = (("ENVI hdr files","*.hdr"),("all files","*.*")))
            if filename3 == '':
                allset = False
            
            if allset:
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
                    
                filename2 = self.rhofilelist[0][0]
                # get the handles of the intercept (rho) data
                if self.rhofilelist[0][1] != None:
                    # the file has been opened, this should always be the case
                    rhodata = self.rhofilelist[0][1]
                    rhodata_map = self.rhofilelist[0][2]
                    rhonumber = self.rhofilelist[0][3].index( self.rhoband.get() )
                else:
                    # this should actually never happen, but still...
                    print("Warning: intercept (rho) file has not been opened. Strange, this should never happen.")
                    # the file will be opened in p_processing(), band #0 will be used
                    rhonumber = 0
    
                N = int( self.PRIwindow.get().split('x')[-1] ) # window size
                # create thread
                self.thread1 = PRI_thread( self.master, filename1, filename2, filename3, rhonumber, N, hypdata, hypdata_map, rhodata, rhodata_map, self.progressvar_PRI )
                
                # prepare GUI and start thread
                # disable all unnecessary stuff here
                self.button_quit.configure( state=DISABLED )
                self.button_datafile.configure( state=DISABLED )
                self.button_rhofile.configure( state=DISABLED )
                self.button_run.configure( text='Stop' )
                self.progressvar_PRI.set(0) 
                self.thread1.start()
            else:
                print("Things not set up properly, not starting processing")

        
if __name__ == '__main__':
    root = Tk()
    GUI = PRIGUI( root )
    root.mainloop()
    

    