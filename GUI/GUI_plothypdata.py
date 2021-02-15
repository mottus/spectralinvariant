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

# sys.path.append("C:\\Users\\MMATTIM\\OneDrive - Teknologian Tutkimuskeskus VTT\\koodid\\python\\hyperspectral\\AIROBEST")
# sys.path.append("hyperspectral\\AIROBEST")

from spectralinvariant.hypdatatools_img import plot_hyperspectral, plot_singleband, get_wavelength
from spectralinvariant.hypdatatools_utils import *
    
class disp_GUI:
        
    def __init__( self, master ):
        
        self.master = master
        self.fig_hypdata = None
        self.hypdata = None # This contains only the mos recently opened figure. Rather useless
        self.hypdata_map = None
        
        bw = 25 # buttonwidth
        
        # Show the GUI in a Toplevel window instead of Root. 
        # This allows many such programs to be run independently in parallel.
        self.w = Toplevel( master )
        self.w.title("GUI for plotting data")

        self.redband_string = StringVar() # string to set and read option_redband OptionMenu
        self.greenband_string = StringVar() # string to set and read option_greenband OptionMenu
        self.blueband_string = StringVar() # string to set and read option_blueband OptionMenu
        self.monoband_string = StringVar() # string to set and read option_monoband OptionMenu
        self.redband_string.set("Red band")
        self.redband_string.set("Green band")
        self.redband_string.set("Blue band")
        self.redband_string.set("Monochrome band")

        self.textlog = ScrolledText( self.w, height=6 )
        self.textlog.pack( side='bottom' )

        self.frame_rgb = Frame( self.w )
        self.label_red= Label( self.frame_rgb, width=bw, text="Red channel")
        self.label_green= Label( self.frame_rgb, width=bw, text="Green channel")
        self.label_blue= Label( self.frame_rgb, width=bw, text="Blue channel")
        self.label_mono= Label( self.frame_rgb, width=bw, text="Monochrome channel")
        self.option_red = OptionMenu( self.frame_rgb, self.redband_string, '' )
        self.option_red['width'] = bw-5
        self.option_green = OptionMenu( self.frame_rgb, self.greenband_string, '' )
        self.option_green['width'] = bw-5
        self.option_blue = OptionMenu( self.frame_rgb, self.blueband_string, '' )
        self.option_blue['width'] = bw-5
        self.option_mono = OptionMenu( self.frame_rgb, self.monoband_string, '' )
        self.option_mono['width'] = bw-5
        self.label_red.pack( side='top' )
        self.option_red.pack( side='top' )
        self.option_red.configure( state=DISABLED )
        self.label_green.pack(side='top')
        self.option_green.pack( side='top' )
        self.option_green.configure( state=DISABLED )
        self.label_blue.pack(side='top')
        self.option_blue.pack( side='top' )
        self.option_blue.configure( state=DISABLED )
        self.label_mono.pack(side='top')
        self.option_mono.pack( side='top' )
        self.option_mono.configure( state=DISABLED )
        self.frame_rgb.pack( side='right' )

        self.frame_button = Frame( self.w )
        self.button_quit = Button( self.frame_button, width=bw, text='Quit', command=self.buttonquit )
        self.button_loaddata = Button( self.frame_button, width=bw, text='Load raster file', command=self.loaddatafile )
        self.button_plottrue = Button( self.frame_button, width=bw, text='Plot truecolor', command=self.plottrue, state=DISABLED )
        self.button_plotmono = Button( self.frame_button, width=bw, text='Plot monochrome', command=self.plotmono, state=DISABLED )
        self.button_plotnir = Button( self.frame_button, width=bw, text='Plot falsecolor NIR', command=self.plotnir, state=DISABLED )
        self.button_plotrgb = Button( self.frame_button, width=bw, text='Plot with three bands', command=self.plotrgb, state=DISABLED )
        self.button_loaddata.pack( side='top' )
        self.button_plottrue.pack( side='top' )
        self.button_plotnir.pack( side='top' )
        self.button_plotrgb.pack( side='top' )
        self.button_plotmono.pack( side='top' )
        self.button_quit.pack( side='bottom' )
        self.frame_button.pack( side='left' )

        # load paths at the end of init (so messaging already exists)
        # self.foldername1 = 'D:\\mmattim\\wrk\\hyytiala-D\\' # where the data is. This is the initial value, will be modified later
        self.foldername1 = get_hyperspectral_datafolder( localprintcommand = self.printlog )

        
    def loaddatafile( self ):
        """
        Open the hyperspectral file.
        """
        self.hypfilename =  filedialog.askopenfilename(initialdir=self.foldername1, title="Choose a hyperspectral data file", filetypes=(("Envi hdr files","*.hdr"),("all files","*.*")))
        
        if self.hypfilename != '':
            self.foldername1 = os.path.split(self.hypfilename)[0]
            self.hypdata_map = None
            self.hypdata = None
            self.button_plotmono.configure( state=DISABLED )
            self.button_plotrgb.configure( state=DISABLED )
            self.button_plotnir.configure( state=DISABLED )
            self.button_plottrue.configure( state=DISABLED )

            # try to have .hdr extension, although this should not be compulsory. No error checking here.
            if not self.hypfilename.endswith(".hdr"):
                self.hypfilename += '.hdr'
                
            # open the files and assign handles
            self.hypdata = spectral.open_image( self.hypfilename )
            self.hypdata_map = self.hypdata.open_memmap()
            
            # come up with band names
            wl_hyp, wl_found = get_wavelength( self.hypfilename, self.hypdata )
            #  best possible result "number:wavelength"
            bandnames = [ '%3d :%6.1f nm' % (i+1,wli) for i,wli in enumerate(wl_hyp) ]
            if wl_found:
                self.printlog("loaddatafile(): Found wavelength information in file "+self.hypfilename+".\n")
            else:
                self.printlog("loaddatafile(): No wavelength information in file "+self.hypfilename+".\n")
                # try to use band name information
                if 'band names' in self.hypdata.metadata:
                    bn = self.hypdata.metadata[ 'band names' ]
                    bandnames = [ '%3d:%s' % (i+1,wli) for i,wli in enumerate(bn) ]
                
            # fill the option menus with wavelengths
            self.option_red['menu'].delete( 0, END )
            self.option_green['menu'].delete( 0, END )
            self.option_blue['menu'].delete( 0, END )
            self.option_mono['menu'].delete( 0, END )
            for choice_num in bandnames:
                choice = str(choice_num)
                self.option_red['menu'].add_command(label=choice, command=lambda v=choice: self.redband_string.set(v))
                self.option_green['menu'].add_command(label=choice, command=lambda v=choice: self.greenband_string.set(v))
                self.option_blue['menu'].add_command(label=choice, command=lambda v=choice: self.blueband_string.set(v))
                self.option_mono['menu'].add_command(label=choice, command=lambda v=choice: self.monoband_string.set(v))
            
            # make reasonable preselections for r,g,b
            if 'default bands' in self.hypdata.metadata:
                if len( self.hypdata.metadata['default bands'] ) > 2:
                    i_r = int( self.hypdata.metadata['default bands'][0] ) - 1
                    i_g = int( self.hypdata.metadata['default bands'][1] ) - 1
                    i_b = int( self.hypdata.metadata['default bands'][2] ) - 1
                    # avoid official printing band names, they usually contain long crappy strings
                    self.printlog( "loaddatafile(): Found default bands (%i,%i,%i) for plotting.\n" % ( i_r, i_g, i_b) )
                else:
                    i_m = int( self.hypdata.metadata['default bands'][0] )  - 1
                    i_r = i_m
                    i_g = i_m
                    i_b = i_m
                    # avoid official printing band names, they usually contain long crappy strings
                    self.printlog( "loaddatafile(): Found one default band (%i) for plotting.\n" % i_m )
                    
            elif wl_found:
                i_r =  abs(wl_hyp-680).argmin() # red band
                i_g =  abs(wl_hyp-550).argmin() # green 
                i_b =  abs(wl_hyp-450).argmin() # blue
            else:
                # just use the first one or three bands
                if self.hypdata_map.shape[2] > 2:
                    # we have at least 3 bands
                    i_r = 0
                    i_g = 1
                    i_b = 2
                else:
                    # monochromatic, use first band only
                    i_r = 0
                    i_g = 0
                    i_b = 0
            # set monochrome to red
            i_m = i_r
            
            # set the optionmenus to their respective values
            self.redband_string.set( bandnames[i_r] )
            self.greenband_string.set( bandnames[i_g] )
            self.blueband_string.set( bandnames[i_b] )
            self.monoband_string.set( bandnames[i_m] )
            
            # wrap it up. Make sure all options are active and ready
            self.option_red.configure( state=ACTIVE )
            self.option_green.configure( state=ACTIVE )
            self.option_blue.configure( state=ACTIVE )
            self.option_mono.configure( state=ACTIVE )                    
            self.button_plotmono.configure( state=ACTIVE )
            self.button_plotrgb.configure( state=ACTIVE )
            if wl_found:
                self.button_plotnir.configure( state=ACTIVE )
                self.button_plottrue.configure( state=ACTIVE )
            
        else:
            self.printlog("loaddatafile(): No file name given.\n")
            
    def plotrgb(self):
        """
        plot in true color, ignore r,g,b band optionmenus
        """
        i_r = int( self.redband_string.get().split(':')[0] )-1
        i_g = int( self.greenband_string.get().split(':')[0] )-1
        i_b = int( self.blueband_string.get().split(':')[0] )-1
        
        self.printlog( "plotrgb(): bands %3d,%3d,%3d.\n" % ( i_r, i_g, i_b ) )
        
        self.fig_hypdata = plot_hyperspectral( self.hypfilename, self.hypdata, self.hypdata_map, self.printlog, plotbands=[i_r,i_g,i_b]  )

            
    def plottrue(self):
        """
        plot using the r,g,b band optionmenus
        """
        wl_hyp, wl_found = get_wavelength( self.hypfilename, self.hypdata )
        if wl_found:
            i_r =  abs(wl_hyp-680).argmin() # red band
            i_g =  abs(wl_hyp-550).argmin() # green 
            i_b =  abs(wl_hyp-450).argmin() # blue
            self.printlog( "plottrue(): %5.1f,%5.1f,%5.1f nm.\n" % (wl_hyp[i_r], wl_hyp[i_g], wl_hyp[i_b]) )
            self.fig_hypdata = plot_hyperspectral( self.hypfilename, self.hypdata, self.hypdata_map, self.printlog, plotbands=[i_r,i_g,i_b] )
        else:
            # this should never happen, but just in case
            self.printlog( "plottrue(): No wavelength data available. This should never happen.\n" )


    def plotnir(self):
        """
        plot in falsecolor NIR, ignore r,g,b band optionmenus
        """
        wl_hyp, wl_found = get_wavelength( self.hypfilename, self.hypdata )
        if wl_found:
            i_r =  abs(wl_hyp-780).argmin() # NIR band
            i_g =  abs(wl_hyp-680).argmin() # red band
            i_b =  abs(wl_hyp-550).argmin() # green 
            self.printlog( "plotnir(): %5.1f,%5.1f,%5.1f nm.\n" % (wl_hyp[i_r], wl_hyp[i_g], wl_hyp[i_b]) )
            self.fig_hypdata = plot_hyperspectral( self.hypfilename, self.hypdata, self.hypdata_map, self.printlog, plotbands=[i_r,i_g,i_b] )
        else:
            # this should never happen, but just in case
            self.printlog( "plotnir(): No wavelength data available. This should never happen.\n" )      
    def plotmono(self):
        """
        Plot the data with the band specified in the monochrome option menu
        """
        i_m = int( self.monoband_string.get().split(':')[0] )-1
        plot_singleband( self.hypfilename, self.hypdata, self.hypdata_map, i_m, self.printlog )
        
            
    def printlog( self , text ):
        """
        Output to log window. Note: no newline added beteen inputs.
        text need not be a string, will be converted when printing.
        """
        self.textlog.insert( END, str(text) )
        self.textlog.yview( END )
        self.master.update_idletasks()

    def buttonquit(self):
        """
        function to end the misery
        note: the pyplot windows are not closed. Maybe, it would be nice to keep track of those to close them
        """
        set_hyperspectral_datafolder( self.foldername1 )
        self.master.destroy() # destruction of root required for program to continue (to its end)
        
if __name__ == '__main__':
    root = Tk()
    GUI = disp_GUI( root )
    root.withdraw()
    root.mainloop()