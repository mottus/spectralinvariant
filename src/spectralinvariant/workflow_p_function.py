# importing all the necessary libraries

from re import findall
from pathlib import Path
from os import cpu_count, path
from time import process_time, time, sleep
from joblib import Parallel, delayed
import numpy as np
import timeit

from spectral import envi
from spectralinvariant.inversion import PROSPECT_D, pc_fast, minimize_cab, golden_cab
from spectralinvariant.spectralinvariants import p,  pC, referencealbedo_transformed, reference_wavelengths
from spectralinvariant.hypdatatools_img import create_raster_like, get_wavelength


def workflow_using_p_function(hypfile_name, file_path, output_filename, chunk_size):
    """
    This function computes p() for a large (or any other) hypdata using the functions in hypdatatools_img.py module
    The input file is accessed using memmap. A raster of the same dimension is created and accessed likewise. Based
    on the user defined chunk size `p()` function is computed for the given data. 
    
    The results of the p() computation i.e. p, rho, DASF and R are stored in seperate column of the outputfile
    for example, output_file_name[:,:, 0:4]

    """
    
    np.seterr(all="ignore")


    def find_nearest(array, value):
        """Finds the index of array element closest to a given value
        """
        idx = (np.abs(array - value)).argmin()
        return idx
    
    file_path = Path(file_path)
    if (file_path/hypfile_name).exists():
        print("Valid file name and file path.")
    else:
        print("Please check file name and or file path!")
        
    
    # Reading the file as a numpy image
    img = envi.open(file_path/hypfile_name)
    
    wavelength = get_wavelength(f'{str(file_path)}\{hypfile_name}')[0]
        
    input_image = img.open_memmap()
    
    # reading metadata
    bands = bands = input_image.shape[2]
    interleave_type = img.__dict__['metadata']['interleave']
   
    scale_factor = img.__dict__['scale_factor']
    scale_factor = int(scale_factor)
    
    
    # creating a raster like file using create_raster_like function
    outdata = create_raster_like(img, output_filename, Nlayers=bands, interleave=interleave_type, outtype=4, force=True)
    outdata_map = outdata.open_memmap(writable=True)
    
    # Preparing inputs for p function (710 >= lambda <= 790)
    # Clipping image and reference spectra in the respective wavelengths 

    b1_p = find_nearest(wavelength, 710)
    b2 = find_nearest(wavelength, 790)

    wls_p = wavelength[b1_p:b2]        
    ref_spectra_p = np.interp(wavelength[b1_p:b2+1], reference_wavelengths(), 0.5*referencealbedo_transformed())
    
    # Computing spectral invariants using p() function
    
    # Get the dimensions of the data
    num_rows, num_cols, num_bands = input_image.shape

    # Define the chunk size 
    line = range(0, num_rows, chunk_size)
   
    start_1 = time()
    start = start_1
    print(f"chunk size = {chunk_size}")

    for i in range(len(line)):
        chunk = input_image[line[i]:line[i] + chunk_size, :, b1_p:b2+1]

        float_chunk = chunk.astype('float')
        nonzero_indices = float_chunk != 0
        float_chunk[nonzero_indices] /= scale_factor
        processed_chunk = p(float_chunk[:, :, :], ref_spectra_p)

        outdata_map[line[i]:line[i] + chunk_size, :, 0:4] = np.transpose(processed_chunk, (1,2,0))

        if i > 0 and i % 10 == 0:
            print(f'Line: {line[i]} / {num_cols}\nComputation time = {round((time()-start), 2)} secs.')
            start = time()
        else:continue

    outdata_map.flush() # writes and saves in the disk

    print()
    print(f"Process completed.\nComputation time = {round((time()-start_1)/60, 2)} mins.")
    
