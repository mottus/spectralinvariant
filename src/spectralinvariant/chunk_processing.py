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
    Wraps p() function from hypdatatools_img.py module to process very large (or any other) file.
    The input file is processed by splitting into chunks defined.
    The input file is accessed using memmap. A raster of the same dimension is created and accessed likewise.
    Based on the chunk size defined `p()` function is computed for the given data. 
    
    Args:
        hypfile_name: ENVI header file,
        input_file_path: file path
        output_filename: file name for processed data. The results of the p() computation i.e. p, rho, DASF and R 
                        are stored as layers in the outputfile
        chunk_size: number of spectra in each each chunk (int value). Defaults value = 3906250
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
    
    wavelength = get_wavelength( fullfilename )[0] # get_wavelength returns a tuple
    
    if wl_idx is None:
        b1_p = (np.abs( wavelength-710) ).argmin()
        b2_p = (np.abs( wavelength-790) ).argmin()
        wl_idx = np.arange( b1_p, b2_p+1 )

    input_image = img.open_memmap()
    
    # Reads metadata of input
    scale_factor = img.__dict__['scale_factor']
    # scale factor not present or missing 
    if scale_factor == None or scale_factor <= 1:
        scale_factor = 10000
    
    # creating a raster like file using create_raster_like function
    output_layer_names = { "layer names": ("p", "intercept", "DASF", "R2") }
    num_output_layers = len(output_layer_names['layer names'])

    description = "Spectral invariants computed for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."
    
    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_layers, metadata_add=output_layer_names, interleave='bip', outtype=4, force=True)
    
    outdata_raster = outdata.open_memmap(writable=True)
    
    # Preparing inputs for p function (710 >= lambda <= 790)
    # Clipping image and reference spectra in the respective wavelengths 
    ref_spectra_p = np.interp(wavelength[ wl_idx ], reference_wavelengths(), 0.5*referencealbedo_transformed())
        
    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols
    
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
        data = input_image_linear[chunk[i]:chunk[i] + chunk_size, :].astype()
        data /= scale_factor
        
        # Computes spectral invariants using p() function
        processed_chunk = p(data[:, :], ref_spectra_p)
        outdata_raster[chunk[i]:chunk[i] + chunk_size, :num_output_layers] = np.transpose(processed_chunk, (1, 0))

    # Converts the output data back to 3D shape
    outdata_raster = outdata_raster.reshape(num_rows, num_cols, num_output_layers)
    outdata_raster.flush() # writes and saves in the disk

    print()
    print(f"{functionname}: computing the spectral invariants completed.\nComputation time = {(process_time()-start)/60: .2f} mins.")
    return 0

def chunk_processing_pC(hypfile_name, file_path, output_filename, chunk_size):
    """
    Wraps pC() function from hypdatatools_img.py module to process very large (or any other) file.
    The input file is processed by splitting into chunks defined.
    The input file is accessed using memmap. A raster of the same dimension is created and accessed likewise.
    Based on the chunk size defined `pC()` function is computed for the given data. 
    
    Args:
        hypfile_name: ENVI header file,
        file_path: file path
        output_filename: file name for processed data
        chunk_size: size of each chunk (int value)
         
    Result:
    The results of the pC() computation i.e. p, rho, DASF,`$R^2$` and C are stored in seperate column of the outputfile
    for example, output_file_name[:, :, 0:4]

    """
    np.seterr(all="ignore")
       
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
    
    # Preparing inputs for pC function (670 >= lambda <= 790)
    b1_p = (np.abs( wavelength-670) ).argmin()
    b2_p = (np.abs( wavelength-790) ).argmin()

       
    ref_spectra_pC = np.interp(wavelength[b1_pC:b2_p+1], reference_wavelengths(), 0.5*referencealbedo_transformed())
           
    # Dimensions of the data
    num_rows, num_cols, num_bands = input_image.shape

    # Define the chunk size 
    line = range(0, num_rows, chunk_size)
    
    print()
    print(f"chunk size = {chunk_size}")
    # Computing spectral invariants using pC() function
    
    start_1 = time()
    start = start_1
    for i in range(len(line)):
        chunk = input_image[line[i]:line[i] + chunk_size, :, b1_pC:b2_p+1]
        
        if np.any(chunk!=0):            
            float_chunk = chunk.astype('float')
            nonzero_indices = float_chunk != 0
            float_chunk[nonzero_indices] /= scale_factor
        else:
            float_chunk = chunk.astype('float')

        processed_chunk = pC(float_chunk[:, :, :], ref_spectra_pC)

        outdata_map[line[i]:line[i] + chunk_size, :, 0:5] = np.transpose(processed_chunk, (1,2,0))

        if i > 0 and i % 10 == 0:
            print(f'Line: {line[i]} / {num_cols}\nComputation time = {round((time()-start), 2)} secs.')
            start = time()
        else:
            continue

    outdata_map.flush() # writes and saves in the disk

    print()
    print(f"Process completed.\nComputation time = {round((time()-start_1)/60, 2)} mins.")


def chunk_processing_chlorophyll(hypfile_name, file_path, output_filename, chunk_size):
    """
    Wraps golden_cab function from inversion.py module to compute chlorophyll content from a given hyperspectral data.
    
    Args:
        hypfile_name: ENVI header file,
        file_path: file path
        output_filename: file name for storing processed data in .npy format
        chunk_size: size of each chunk (int value)
         
    Result:
    The results of the golden_cab computation i.e. chlorophyll content is stored as a '.npy' file
    """
    np.seterr(all="ignore")
       
    file_path = Path(file_path)
    if (file_path/hypfile_name).exists():
        print("Valid file name and file path.")
    else:
        print("Please check file name and or file path!")
        
    # Reading the file as a numpy image
    img = envi.open(file_path/hypfile_name)
    
    wavelength = get_wavelength(f'{str(file_path)}\{hypfile_name}')[0]
        
    input_image = img.open_memmap()
    
    band1_cab = (np.abs( wavelength-670) ).argmin()
    band2_cab = (np.abs( wavelength-720) ).argmin()
    wavelength_subset = wavelength[band1_cab:band2_cab+1]

    # Creating an instance of the PROSPECT class with the input values specified by Ihalainen et al. (2023)
    model = PROSPECT_D(N=1.5, Car=1.0, Cw=0.0, Cm=0.0)
    model.subset(wavelength_subset)

    num_rows, num_cols, num_bands = (input_image[:, :, band1_cab:band2_cab+1]).shape
    num_idx = num_rows*num_cols
    input_image_linear = input_image[:, :, band1_cab:band2_cab+1].reshape(num_idx, num_bands)

    # chunk_size = np.round( 0.5e9 / 128).astype(int)
    # chunk_size = 100
    chunk_idx = np.arange(0, num_idx, chunk_size)
    inversion_result = []

    start = time()

    for i in range(len(chunk_idx)): 
        
        chunk = input_image_linear[chunk_idx[i]:chunk_idx[i]+chunk_size, :]
        
        if np.any(chunk!=0):   
            float_chunk = chunk.astype('float')            
            float_chunk /= 10000 #scale_factor             
        else:
            float_chunk = chunk.astype('float')
            
        start_1 = time()
        
        inversion_result[chunk_idx[i]:chunk_idx[i]+chunk_size] = Parallel(n_jobs=cpu_count())(delayed(golden_cab)(
        model, pixel, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, bounds=(1., 100.)) for pixel in float_chunk)
        
        if i > 0 and i % (chunk_size*10) == 0:
            print(f'Chunk: {i} / {len(chunk_idx)}.')
            print(f'computation took: {(time() - start_1):.2f} seconds.')
            print()

    end = time()
    print(f'Finished! The inversion took {(end - start)/60:.2f} mins.')

    cabs = np.array(inversion_result)
    cabs = cabs.reshape(num_rows, num_cols)
    np.save(output_filename+'.npy', cabs)


