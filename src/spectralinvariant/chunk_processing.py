# importing all the necessary libraries

from re import findall
from pathlib import Path
from time import process_time, time, sleep
from joblib import Parallel, delayed
import numpy as np
import os

from spectral import envi
from spectralinvariant.inversion import PROSPECT_D, pc_fast, minimize_cab, golden_cab
from spectralinvariant.spectralinvariants import p,  pC, referencealbedo_transformed, reference_wavelengths
from spectralinvariant.hypdatatools_img import create_raster_like, get_wavelength


def chunk_processing_p(hypfile_name, input_file_path, output_filename, chunk_size=None, wl_idx=None):
    """
    Wraps p() function from hypdatatools_img.py module to process hyperspectral data.
    The input file is processed in chunks. A raster of the same spatial dimension with 4 layers is created to store the results in consecutive layers.
    
    Args:
        hypfile_name: ENVI header file,
        input_file_path: file path
        output_filename: file name for processed data. The results of the p() computation i.e. p, rho, DASF and R 
                        are stored as layers in the outputfile
        chunk_size: number of spectra in each each chunk (int value). Default value = 3906250
        wl_idx: index of wavelengths used in computations. Defaults to 710-790 nm
         
    Result:
        returns 0 if the compuation is successfull, else -1.       
    """

    np.seterr(all="ignore")
    functionname = "chunk_processing_p()"
    fullfilename = os.path.join(input_file_path, hypfile_name)

    if not os.path.exists(fullfilename):
        print(functionname + " ERROR: file " + fullfilename + "does not exist")
        return -1

    img = envi.open( fullfilename )
    input_image = img.open_memmap()

    wavelength = get_wavelength( fullfilename )[0] # get_wavelength returns a tuple

    if wl_idx is None:
        b1_p = (np.abs( wavelength-710) ).argmin()
        b2_p = (np.abs( wavelength-790) ).argmin()
        wl_idx = np.arange( b1_p, b2_p+1 )

    # Read metadata of input
    try:
        scale_factor = img.__dict__['metadata']['reflectance scale factor'].astype(float)        
    except:
        scale_factor = 10000.0 # scale factor missing from metadata

    # creating a raster like file using create_raster_like function
    output_layer_names = { "layer names": ("p", "intercept", "DASF", "R2") }
    num_output_layers = len(output_layer_names['layer names'])

    description = "Spectral invariants computed for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_layers, metadata_add=output_layer_names, interleave='bip', outtype=4, force=True)

    outdata_raster = outdata.open_memmap(writable=True)

    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols

    # Preparing inputs for p function (710 >= lambda <= 790)
    ref_spectra_p = np.interp(wavelength[ wl_idx ], reference_wavelengths(), 0.5*referencealbedo_transformed())

    # Converts input and output data into 2D shape
    input_image_linear = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].reshape(num_idx, num_bands)
    outdata_raster = outdata_raster.reshape(num_idx, num_output_layers)

    # Defines the chunk size
    if chunk_size == None:        
        chunk_size = np.round( 0.5e9 / 128).astype(int)

    chunk = range(0, num_idx, chunk_size)

    print()
    print("Please wait! Processing the data ....")

    start = process_time()

    for i in range(len(chunk)):
        data = input_image_linear[chunk[i]:chunk[i] + chunk_size, :]
        data_float = data.copy().astype('float')
        data_float /= scale_factor
        
        # Computes spectral invariants using p() function
        processed_chunk = p(data_float[:, :], ref_spectra_p)
        outdata_raster[chunk[i]:chunk[i] + chunk_size, :num_output_layers] = np.transpose(processed_chunk, (1, 0))

    # Converts the output data back to 3D shape
    outdata_raster = outdata_raster.reshape(num_rows, num_cols, num_output_layers)
    outdata_raster.flush() # writes and saves in the disk

    print()
    print(f"{functionname}: computing the spectral invariants completed.\nComputation time = {(process_time()-start)/60: .2f} mins.")
    return 0

def chunk_processing_pC(hypfile_name, input_file_path, output_filename, chunk_size=None, wl_idx=None):
    """
            
    Wraps pC() function from hypdatatools_img.py module to process hyperspectral data.
    The input file is processed in chunks. A raster of the same spatial dimension with 5 layers is created to store the results produced consecutively.
    
    Args:
        hypfile_name: ENVI header file,
        input_file_path: file path
        output_filename: file name for processed data. The results of the pC() computation i.e. p, rho, DASF, R2 and C.
                        are stored as layers in the outputfile
        chunk_size: number of spectra in each each chunk (int value). Default value = 3906250
        wl_idx: index of wavelengths used in computations. Defaults to 670-790 nm
         
    Result:
        returns 0 if the compuation is successfull, else -1.  

    """
    np.seterr(all="ignore")
    functionname = "chunk_processing_pC()"
    fullfilename = os.path.join(input_file_path, hypfile_name)

    if not os.path.exists(fullfilename):
        print(functionname + " ERROR: file " + fullfilename + "does not exist")
        return -1

    img = envi.open( fullfilename )
    input_image = img.open_memmap()

    wavelength = get_wavelength( fullfilename )[0] # get_wavelength returns a tuple

    if wl_idx is None:
        b1_p = (np.abs( wavelength-670) ).argmin()
        b2_p = (np.abs( wavelength-790) ).argmin()
        wl_idx = np.arange( b1_p, b2_p+1 )

    # Read metadata of input
    try:
        scale_factor = img.__dict__['metadata']['reflectance scale factor'].astype(float)        
    except:
        scale_factor = 10000.0 # scale factor missing from metadata

    # creating a raster like file using create_raster_like function
    output_layer_names = { "layer names": ("p", "intercept", "C", "DASF", "R2") }
    num_output_layers = len(output_layer_names['layer names'])

    description = "Spectral invariants computed for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_layers, metadata_add=output_layer_names, interleave='bip', outtype=4, force=True)

    outdata_raster = outdata.open_memmap(writable=True)

    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols

    # Preparing inputs for p function (710 >= lambda <= 790)
    ref_spectra_p = np.interp(wavelength[ wl_idx ], reference_wavelengths(), 0.5*referencealbedo_transformed())

    # Converts input and output data into 2D shape
    input_image_linear = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].reshape(num_idx, num_bands)
    outdata_raster = outdata_raster.reshape(num_idx, num_output_layers)

    # Defines the chunk size
    if chunk_size == None:        
        chunk_size = np.round( 0.5e9 / 128).astype(int)

    chunk = range(0, num_idx, chunk_size)

    print()
    print("Please wait! Processing the data ....")

    start = process_time()

    for i in range(len(chunk)):
        data = input_image_linear[chunk[i]:chunk[i] + chunk_size, :]
        data_float = data.copy().astype('float')
        data_float /= scale_factor
        
        # Computes spectral invariants using p() function
        processed_chunk = pC(data_float[:, :], ref_spectra_p)
        outdata_raster[chunk[i]:chunk[i] + chunk_size, :num_output_layers] = np.transpose(processed_chunk, (1, 0))

    # Converts the output data back to 3D shape
    outdata_raster = outdata_raster.reshape(num_rows, num_cols, num_output_layers)
    outdata_raster.flush() # writes and saves in the disk

    print()
    print(f"{functionname}: computing the spectral invariants completed.\nComputation time = {(process_time()-start)/60: .2f} mins.")
    return 0


def chunk_processing_chlorophyll(hypfile_name, hypfile_path, output_filename, chunk_size=None, wl_idx=None):

    """
    Wraps golden_cab function from inversion.py module on propspect_D model to compute chlorophyll content for a given hyperspectral data.
    A raster of the same spatial dimension is created to store the result.
    
    Args:
        hypfile_name: ENVI header file,
        file_path: file path
        output_filename: file name for storing processed data in .npy format
        chunk_size: size of each chunk (int value). Default value = 3906250
        wl_idx: index of wavelengths used in computations. Defaults to 670-720 nm
         
    Result:
        returns 0 if the compuation is successfull, else -1.

    """

    np.seterr(all="ignore")
    functionname = "chunk_processing_chlorophyll()"
    fullfilename = os.path.join(hypfile_path, hypfile_name)

    if not os.path.exists(fullfilename):
        print(functionname + " ERROR: file " + fullfilename + "does not exist")
        return -1

    img = envi.open( fullfilename )
    input_image = img.open_memmap()

    wavelength = get_wavelength( fullfilename )[0] # get_wavelength returns a tuple

    if wl_idx is None:
        b1_cab = (np.abs( wavelength-670) ).argmin()
        b2_cab = (np.abs( wavelength-720) ).argmin()
        wl_idx = np.arange( b1_cab, b2_cab + 1 )

    # Read metadata of input
    try:
        scale_factor = img.__dict__['metadata']['reflectance scale factor'].astype(float)        
    except:
        scale_factor = 10000.0 # scale factor missing from metadata

    # creating a raster like file using create_raster_like function
    output_layer_names = { "layer names": ("chlorophyll content",) } # !! not "," == 19
    num_output_layers = len(output_layer_names['layer names'])

    description = "Chlorophyll computed for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_layers, metadata_add=output_layer_names, interleave='bip', outtype=4, force=True)

    outdata_raster = outdata.open_memmap(writable=True)

    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols
   
    input_image_linear = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].reshape(num_idx, num_bands)
    
    # convert the raster to 1D array
    outdata_raster = outdata_raster.reshape(num_idx)    
    
    if chunk_size == None:
        chunk_size = np.round( 0.5e9 / 128).astype(int)
    
    chunk = np.arange(0, num_idx, chunk_size)

    # Creating an instance of the PROSPECT class with the input values specified by Ihalainen et al. (2023)
    model = PROSPECT_D(N=1.5, Car=1.0, Cw=0.0, Cm=0.0)
    model.subset(wavelength[wl_idx])
    
    print()
    print("Please wait! Processing the data ....")
    start = time()

    for i in range(len(chunk)):
        start1 = time()
        data = input_image_linear[chunk[i]:chunk[i]+chunk_size, :]
        data_float = data.astype('float')            
        data_float /= scale_factor # reflectance scale factor             
        
        #               
        outdata_raster[chunk[i]:chunk[i]+chunk_size] = Parallel(n_jobs=cpu_count())(delayed(golden_cab)(
        model, pixel, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, bounds=(1., 100.)) for pixel in data_float)
        
        print()
        print(f'Chunk :{i} / {len(chunk)}\nComputation time = {(time()-start1)/60:.2f} mins.')
    # convert the raster back to 2D shape        
    outdata_raster = outdata_raster.reshape(num_rows, num_cols)
    outdata_raster.flush() # writes and saves in the disk
    print()
    print(f'Cholorphyll map compuation completed!\nCompuation time = {(time() - start)/60:.2f} mins.')
    return 1

