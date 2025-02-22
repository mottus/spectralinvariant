# importing all the necessary libraries

from re import findall
from pathlib import Path
from time import process_time, time, sleep
from joblib import Parallel, delayed
import numpy as np
import os
from os import cpu_count

from spectral import envi
from spectralinvariant.inversion import PROSPECT_D, pc_fast, minimize_cab, golden_cab
from spectralinvariant.spectralinvariants import compute_p,  compute_pC, referencealbedo_transformed, reference_wavelengths
from spectralinvariant.hypdatatools_img import create_raster_like, get_wavelength, get_scalefactor

def chunk_processing_p(hypfile_name, hypfile_path=None, output_filename="p_data", chunk_size=None, wl_range=[710,790]):
    """ Wraps p() function from spectralinvariants.py module to process hyperspectral data.
    
    Uses the transformed reference leaf albedo. The input file is processed in chunks.
    A raster of the same spatial dimension with 4 layers is created to store the results consecutively.
    
    Args:
        hypfile_name: ENVI header file,
        hypfile_path: Envi file path (optional)
        output_filename: Envi file name (absolute) for processed data: i.e. p, rho, DASF and R 
            if None, default name in current working directory is used (NOTE: this may change to a more reasonable default!)
        chunk_size: number of spectra in each each chunk (int value). Default value = 3906250 spectra
        wl_range: a list with the start and end wavelengths for analyses
         
    Result:
        returns 0 if the compuation is successful, else -1.
    """

    np.seterr(all="ignore")
    functionname = "chunk_processing_p()"
    if hypfile_path is None:
        fullfilename = hypfile_name
    else:
        fullfilename = os.path.join(hypfile_path, hypfile_name)

    if not os.path.exists(fullfilename):
        print(functionname + " ERROR: file " + fullfilename + "does not exist!")
        return -1

    img = envi.open( fullfilename )
    input_image = img.open_memmap()

    wavelength, wl_found = get_wavelength( fullfilename )
    if not wl_found:
        print(functionname + " ERROR: wavelength information not found in " + fullfilename )
        return -1

    # sort the waelengths in ascending order
    wl_range.sort()

    # check the values of wl_idx are within the range of wavelength
    if not ( wavelength[0] <= wl_range[0] and wavelength[-1] >= wl_range[-1] ):
        print(functionname + " ERROR: wavelength indices do not exist")
        return -1

    # find the location of the nearest wl in the array
    b1_p = (np.abs(wavelength - wl_range[0])).argmin() 
    b2_p = (np.abs(wavelength - wl_range[1])).argmin()
    wl_idx = np.arange( b1_p, b2_p+1 )
   
    # Define the chunk size
    if chunk_size == None:        
        chunk_size = np.round( 0.5e9 / 128).astype(int)

    scale_factor = get_scalefactor( fullfilename )
    
    # creating a raster like file using create_raster_like function
    output_band_names = { "band names": ("p", "intercept", "DASF", "R2") }
    num_output_bands = len(output_band_names['band names'])

    description = "Spectral invariants computed with transformed reference leaf albedo for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_bands, metadata_add=output_band_names, interleave='bip', outtype=4, force=True)

    outdata_map = outdata.open_memmap(writable=True)

    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols

    # Preparing inputs for p function (710 >= lambda <= 790)
    ref_spectra_p = np.interp(wavelength[ wl_idx ], reference_wavelengths(), referencealbedo_transformed())

    # Converts input and output data into 2D shape
    input_image_linear = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].reshape(num_idx, num_bands)
    outdata_map = outdata_map.reshape(num_idx, num_output_bands)

    print("\nPlease wait! Processing the data...")
    chunk = range(0, num_idx, chunk_size)
    start = process_time()
    for i in range(len(chunk)):
        data = input_image_linear[chunk[i]:chunk[i] + chunk_size, :]
        data_float = data.astype('float')
        data_float /= scale_factor
        
        # Computes spectral invariants using p() function
        processed_chunk = compute_p(data_float[:, :], ref_spectra_p)
        outdata_map[chunk[i]:chunk[i] + chunk_size, :num_output_bands] = np.transpose(processed_chunk, (1, 0))

    # Converts the output data back to 3D shape
    outdata_map = outdata_map.reshape(num_rows, num_cols, num_output_bands)
    outdata_map.flush() # writes and saves in the disk

    print()
    print(f"{functionname}: computing the spectral invariants completed.\nComputation time = {(process_time()-start)/60: .2f} mins.")
    return 0

def chunk_processing_pC(hypfile_name, hypfile_path=None, output_filename="pC_data", chunk_size=None, wl_range=[670, 790]):
    """ Wraps pC() function from spectralinvariants.py module to process hyperspectral data.
    
    Uses the transformed reference leaf albedo. 
    The input file is processed in chunks. A raster of the same spatial dimension with 5 layers is created to store the results produced consecutively.
    
    Args:
        hypfile_name: ENVI header file,
        hypfile_path: Envi file path (optional)
        output_filename: Envi file name (absolute) for processed data: i.e. p, rho, DASF, R2 and C 
            if None, default name in current working directory is used (NOTE: this may change to a more reasonable default!)
        chunk_size: number of spectra in each each chunk (int value). Default value = 3906250 spectra
        wl_range: a list with the start and end wavelengths for analyses
         
    Result:
        returns 0 if the compuation is successfull, else -1.
    """
    np.seterr(all="ignore")
    functionname = "chunk_processing_pC()"
    if hypfile_path is None:
        fullfilename = hypfile_name
    else:
        fullfilename = os.path.join(hypfile_path, hypfile_name)

    if not os.path.exists(fullfilename):
        print(functionname + " ERROR: file " + fullfilename + "does not exist")
        return -1

    img = envi.open( fullfilename )
    input_image = img.open_memmap()

    wavelength, wl_found = get_wavelength( fullfilename )
    if not wl_found:
        print(functionname + " ERROR: wavelength information not found in " + fullfilename )
        return -1

    # sort the the list in ascending order
    wl_range.sort()

    # check the values of wl_range are within the range of wavelength
    is_wl_idx_valid =  wavelength[0] <= wl_range[0] and wavelength[-1] >= wl_range[-1]
    
    if not is_wl_idx_valid:
        print(functionname + " ERROR: wavelength indices do not exist!")
        return -1

    # find the location of the nearest wl in the array
    b1_pC = (np.abs(wavelength - wl_range[0])).argmin()
    b2_pC = (np.abs(wavelength - wl_range[1])).argmin()  
    wl_idx = np.arange( b1_pC, b2_pC+1 )

    scale_factor = get_scalefactor( fullfilename )

    # Defines the chunk size
    if chunk_size == None:        
        chunk_size = np.round( 0.5e9 / 128).astype(int)

    # creating a raster like file using create_raster_like function
    output_band_names = { "band names": ("p", "intercept", "DASF", "R2", "C") }
    num_output_bands = len(output_band_names['band names'])

    description = "Spectral invariants computed with transformed reference leaf albedo for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_bands, metadata_add=output_band_names, interleave='bip', outtype=4, force=True)

    outdata_map = outdata.open_memmap(writable=True)

    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols

    # Preparing inputs for p function (710 >= lambda <= 790)
    ref_spectra_p = np.interp(wavelength[ wl_idx ], reference_wavelengths(), referencealbedo_transformed())

    # Convert input and output data into 2D shape
    input_image_linear = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].reshape(num_idx, num_bands)
    outdata_map = outdata_map.reshape(num_idx, num_output_bands)

    chunk = range(0, num_idx, chunk_size)
    print("\nPlease wait! Processing the data...")
    start = process_time()
    for i in range(len(chunk)):
        data = input_image_linear[chunk[i]:chunk[i] + chunk_size, :]
        data_float = data.astype('float')
        data_float /= scale_factor
        
        # Compute spectral invariants using p() function
        processed_chunk = compute_pC(data_float[:, :], ref_spectra_p)
        outdata_map[chunk[i]:chunk[i] + chunk_size, :num_output_bands] = np.transpose(processed_chunk, (1, 0))

    # Convert the output data back to 3D shape
    outdata_map = outdata_map.reshape(num_rows, num_cols, num_output_bands)
    outdata_map.flush() # writes and saves in the disk

    print()
    print(f"{functionname}: computing the spectral invariants completed.\nComputation time = {(process_time()-start)/60: .2f} mins.")
    return 0

def create_chlorophyll_map(hypfile_name, hypfile_path=None, output_filename="chl_content_map", chunk_size=None, wl_range=[670, 720], inv_algorithm=None, inv_method=None, **kwargs):

    """
    Wraps golden_cab (`scipy.golden`) and minimize_cab (`scipy.minimize`) algorithms implemented in inversion.py module on propspect_D model to compute chlorophyll content for a given hyperspectral data.
    A raster of the same spatial dimension as the hyperspectral data is created to store the result.   
    
    Args:
        hypfile_name:    ENVI header file,
        hypfile_path:    file path
        output_filename: file name for storing chlorophyll map
        chunk_size: size of each chunk (int value). Default value = 3906250 pixels
        wl_range: a list with the start and end wavelengths for analyses
        inv_algorithm:   integer value, 1 or 2 to select an inversion algorithm for the computation.
                            1 = Goldencab (Default algorithmfor faster computational time)
                            2 = Minimizecab

        inv_method: integer value between 1 to 6 (to select an optimization method, if inv_algorithm == 2)
                        1 = Nelder-Mead (Default method)
                        2 = L-BFGS-B
                        3 = TNC
                        4 = SLSQP
                        5 = Powell
                        6 = trust-constr                                    
        
        **kwargs : dict, keyword arguments for the `scipy.golden` or `scipy.minimize` function
         
    Result:
        output file is stored in the current working directory 
        returns 0 if the compuation is successfull, else -1

    """

    np.seterr(all="ignore")
    functionname = "create_chlorophyll_map()"
    
    if hypfile_path is None:
        fullfilename = hypfile_name
    else:
        fullfilename = os.path.join(hypfile_path, hypfile_name)
    
    if not os.path.exists(fullfilename):
        print(functionname + " ERROR: file " + fullfilename + "does not exist")
        return -1

    img = envi.open( fullfilename )
    input_image = img.open_memmap()

    wavelength, wl_found = get_wavelength( fullfilename )
    if not wl_found:
        print(functionname + " ERROR: wavelength information not found in " + fullfilename )
        return -1

    # sort the the list in ascending order just to be sure
    wl_range.sort()

    # check the values of wl_range are within the range of wavelength
    is_wl_idx_valid =  wavelength[0] <= wl_range[0] and wavelength[-1] >= wl_range[-1]
    
    if not is_wl_idx_valid:
        print(functionname + " ERROR: wavelength indices do not exist !")
        return -1

    # find the location of the nearest wl in the array
    b1_cab = (np.abs(wavelength - wl_range[0])).argmin()
    b2_cab = (np.abs(wavelength - wl_range[1])).argmin()
    wl_idx = np.arange( b1_cab, b2_cab+1 )

    # # Read reflectance scale factor of the hyp file/data
    scale_factor = get_scalefactor( fullfilename )

    # inversion algorithm selection
    algorithm = {1:"golden_cab", 2:"minimize_cab"}
    if inv_algorithm is None:
        inv_algorithm = int(1)
    
    if not (inv_algorithm == int(1) or inv_algorithm == int(2)):
        print(f"{functionname} ERROR: invalid argument passed for 'inv_algorithm' parameter!\nvalid argument = 1 or 2 !")
        return -1
    print(f"Inversion alogrithm : {algorithm[inv_algorithm]}")

    # optimization methods for `scipy.minimize` algorithm
    optimization = {1:'Nelder-Mead', 2:'L-BFGS-B', 3:'TNC', 4:'SLSQP', 5:'Powell', 6:'trust-constr'}

    if inv_algorithm == 2:
        if inv_method is None:
            inv_method = 1
            opt_method = optimization[inv_method] # Default value
        elif inv_method in range(1,7):
            opt_method = optimization[inv_method]
        else:
            print(f"{functionname} ERROR: invalid argument passed for 'inv_method' parameter!\nvalid argument = an integer between 1 to 6 !")
            return -1
        print(f"Optimization method : {optimization[inv_method]}")

    # setting default values of keyword arguments if the values are not passed
    g = kwargs.get('gamma', 1.0)
    p_l = kwargs.get('p_lower', 0.0)
    p_u = kwargs.get('p_upper' ,1.0) 
    r_l = kwargs.get('rho_lower', 0.0)
    r_u = kwargs.get('rho_upper', 2.0)
    b = kwargs.get('bounds', (1., 100.))
    ig = kwargs.get('initial_guess', 30.)
    
    if chunk_size == None:
        chunk_size = np.round( 0.5e9 / 128).astype(int)


    # creating a raster  using 'create_raster_like' function
    output_band_names = { "band names": ("chlorophyll content",) } # !! not "," == 19
    num_output_layers = len(output_band_names['band names'])

    description = "Chlorophyll computed for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata = create_raster_like(img, output_filename, description=description,
        Nlayers=num_output_layers, metadata_add=output_band_names, interleave='bip', outtype=4, force=True)

    outdata_map = outdata.open_memmap(writable=True)

    # Dimensions of the data used in result computation
    num_rows, num_cols, num_bands = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].shape
    num_idx = num_rows * num_cols
   
    input_image_linear = input_image[:, :, wl_idx[0]:wl_idx[-1]+1].reshape(num_idx, num_bands)
    
    # convert the raster to 1D array
    outdata_map = outdata_map.reshape(num_idx)

    # Creating an instance of the PROSPECT class with the input values specified by Ihalainen et al. (2023)
    model = PROSPECT_D(N=1.5, Car=1.0, Cw=0.0, Cm=0.0)
    model.subset(wavelength[wl_idx])
    
    print()
    print("Please wait! Processing the data ....")

    chunk = np.arange(0, num_idx, chunk_size)
    
    start = time()  
    for i in range(len(chunk)):
        start1 = time()
        data = input_image_linear[chunk[i]:chunk[i]+chunk_size, :]
        data_float = data.astype('float')            
        data_float /= scale_factor # reflectance scale factor
        
        if inv_algorithm == int(1):
            outdata_map[chunk[i]:chunk[i]+chunk_size] = Parallel(n_jobs=cpu_count())(delayed(golden_cab)(
                model, pixel, gamma=g, p_lower=p_l, p_upper=p_u, rho_lower=r_l, rho_upper=r_u, bounds=b) for pixel in data_float)
        else:      
            outdata_map[chunk[i]:chunk[i]+chunk_size] = Parallel(n_jobs=cpu_count())(delayed(minimize_cab)(
                model, pixel, method=opt_method, gamma=g, p_lower=p_l, p_upper=p_u, rho_lower=r_l, rho_upper=r_u, initial_guess=ig, bounds=[b]) for pixel in data_float)      

        print()
        print(f'Chunk :{i+1} / {len(chunk)}\nComputation time = {(time()-start1)/60:.2f} mins.')

    # convert the raster back to 2D shape        
    outdata_map = outdata_map.reshape(num_rows, num_cols)
    outdata_map.flush() # writes and saves in the disk
    print()
    print(f'Cholorphyll map compuation completed!\nCompuation time = {(time() - start)/60:.2f} mins.')
    return 0

def compute_inversion_invariants(hypfile_name, cmap_filename="chl_content_map.hdr", hypfile_path=None, cmap_file_path=None, output_filename="chl_inversion_invariants", wl_range=[670, 720]):

    """
    Wraps pc_fast function from inversion.py module on propspect_D model to compute spectral invariants from a chlorophyll map, produced using inversion algorithm.
    The result is stored in a raster of the same spatial dimension with 3 layers in the current working directory.
       
    Args:
        hypfile_name: ENVI header file (hyperspectral data),
        hypfile_path: file path
        cmap_filename: ENVI header file (chlorophyll map of the hypdata created using inversion algorithm)
        cmap_file_path: file path
        
        output_filename_inv: file name to store the results of pc_fast() computation       
                        
        chunk_size: size of each chunk (int value). Default value = 3906250 pixel
        wl_range: a list with the start and end wavelengths for analyses
         
    Result:
        output file with estimated values of rho, p and c  
        returns 0 if the compuation is successfull, else -1

    """
    
    np.seterr(all="ignore")
    functionname = "compute_inversion_invariants()"

    if hypfile_path is None:
        hyp_fullfilename = hypfile_name
    else:
        hyp_fullfilename = os.path.join(hypfile_path, hypfile_name)

    if not os.path.exists(hyp_fullfilename):
        print(functionname + " ERROR: file " + hyp_fullfilename + "does not exist")
        return -1

    if cmap_file_path is None:
        cmap_fullfilename = cmap_filename
    else:
        cmap_fullfilename = os.path.join(cmap_file_path, cmap_filename)
    
    if not os.path.exists(cmap_fullfilename):
        print(functionname + " ERROR: file " + cmap_fullfilename + "does not exist")
        return -1
        
    # Reading hypdata and chlorophyll map  #
    
    hyp_img = envi.open( hyp_fullfilename )
    input_image = hyp_img.open_memmap()
    
    wavelength, wl_found = get_wavelength( hyp_fullfilename )
    if not wl_found:
        print(functionname + " ERROR: wavelength information not found in " + fullfilename )
        return -1

    # sort the the list in ascending order
    wl_range.sort()

    # check the values of wl_idx are within the range of wavelength
    is_wl_idx_valid =  wavelength[0] <= wl_range[0] and wavelength[-1] >= wl_range[-1]
    
    if not is_wl_idx_valid:
        print(functionname + " ERROR: wavelength indices do not exist !")
        return -1

    # find the location of the nearest wl in the array
    b1_p = (np.abs(wavelength - wl_range[0])).argmin()
    b2_p = (np.abs(wavelength - wl_range[1])).argmin()
    wl_idx = np.arange( b1_p, b2_p+1 )

    # Read reflectance scale factor of the hyp file/data
    scale_factor = get_scalefactor( hyp_fullfilename )

    # Dimensions of the hypdata
    num_rows, num_cols, num_bands = input_image[:, :, :].shape
    num_idx = num_rows * num_cols
 
    # Reading chlorophyll map
    cab_img = envi.open(cmap_fullfilename )
    cabs = cab_img.open_memmap()   

    # Creating a raster for saving results from pc_fast()

    # First raster for storing results from pc_fast() i.e, p, rho and C. 
    output_band_names = {"band names": ("rho", "p", "C")}
    num_output_layers = len(output_band_names['band names'])

    description1 = "Inversion result computed using pc_fast() for " + hypfile_name + " "\
        + str( wavelength[wl_idx[0]] ) + "-" + str( wavelength[wl_idx[-1]] ) + " nm."

    outdata_1 = create_raster_like(hyp_img, output_filename, description=description1,
        Nlayers=num_output_layers, metadata_add=output_band_names, interleave='bip', outtype=4, force=True)

    outdata_pc_fast = outdata_1.open_memmap(writable=True)
    
    #  Reshaping input and output files to 2D  

    input_image_linear = input_image[:, :, :].reshape(num_idx, num_bands)
    input_image_subset_linear = input_image_linear[:, wl_idx[0]:wl_idx[-1]+1]
    cabs_linear = cabs.reshape(num_idx)
    
    outdata_pc_fast = outdata_pc_fast.reshape(num_idx, num_output_layers)
       

    # Creating an instance of the PROSPECT class with the input values specified by Ihalainen et al. (2023)
    model = PROSPECT_D(N=1.5, Car=1.0, Cw=0.0, Cm=0.0)
    model.subset(wavelength[wl_idx])
    
    # Computing rho, p, and C
    
    print()
    print("Please wait! computing pc_fast() ....")

    start = time()
    start1 = start 

    for i, pixel_cab in enumerate (input_image_subset_linear):
        
        pixel_cab = pixel_cab.astype('float')
        pixel_cab /= scale_factor
               
        processed_pixel = pc_fast(pixel_cab, model.PROSPECT(Cab=cabs_linear[i]))
        outdata_pc_fast[i, :] = processed_pixel
        
        if i > 0 and i % (num_idx // 10) == 0: # (added for testing)
            print(f'Iteration: {i+1} / {len(input_image_linear)}\nComputation time = {(time()-start1): .2f} seconds.')
            start1 = time()
        
    # converting output back to 3D shape        
    outdata_pc_fast = outdata_pc_fast.reshape(num_rows, num_cols, num_output_layers)
    outdata_pc_fast.flush() # writes and saves in the disk
    print(f"{functionname}: process completed!\nProcess completion time = {(time() - start)/60:.2f} mins.")
    
    return 0

def compute_illumination_corrected_leaf_spectra(hypfile_name, hypfile_path=None, inversion_filename="chl_inversion_invariants.hdr",  inversion_filepath=None, output_filename="illumination_corrected", chunk_size=None):

    """ Computes illumination corrected leaf spectrum using hypdata and spectral invariants (computed from inversion algorithm i.e. `pc_fast()`).
        
    Args:
        hypfile_name: ENVI header file for hyperspectral data
        hypfile_path: file path for the hyperspectral data (optional)
        inversion_filename: name of the header for the Envi file containing rho, p and C
            Defaults to the default file name of chunk_processing_pC() in the current directory
        inversion_path: path for the inversion files (optional)
        output_filename: absolute Envi file name for storing results of corrected leaf reflectance
            NOTE: the default name may change to a more reasonable value in the future
        chunk_size: size of each chunk (int value). Default value = 3906250 pixels
         
    Result:
        output file is stored in the current working directory 
        returns 0 if the compuation is successfull, else -1.
    """
    
    np.seterr(all="ignore")
    functionname = "compute_illumination_corrected_leaf_spectra()"
    
    if hypfile_path is None:
        hyp_fullfilename = hypfile_name
    else:
        hyp_fullfilename = os.path.join(hypfile_path, hypfile_name)
    if not os.path.exists(hyp_fullfilename):
        print(functionname + " ERROR: file " + hyp_fullfilename + "does not exist")
        return -1
    
    if inversion_filepath is None:
        inv_fullfilename = inversion_filename
    else:
        inv_fullfilename = os.path.join(inversion_filepath, inversion_filename)    
    if not os.path.exists(inv_fullfilename):
        print(functionname + " ERROR: file " + inv_fullfilename + "does not exist")
        return -1

    #  Reading hypdata and inversion results  #

    hyp_img = envi.open( hyp_fullfilename )
    input_image = hyp_img.open_memmap()

    wavelength, wl_found = get_wavelength( hyp_fullfilename )
    if not wl_found:
        print(functionname + " ERROR: wavelength information not found in " + hyp_fullfilename )
        return -1
  
    scale_factor = get_scalefactor( hyp_fullfilename )
    s_f = {'reflectance scale factor': scale_factor}

    # Dimensions of the hypdata
    num_rows, num_cols, num_bands = input_image[:, :, :].shape
    num_idx = num_rows * num_cols
 
    # Reading inversion results
    inv_img = envi.open(inv_fullfilename )
    inversion_result = inv_img.open_memmap()
    num_inv_bands = inversion_result.shape[2]

    #  Creating a raster to save corrected leaf reflectance result  #

    description = "Corrected leaf reflectance for " + hypfile_name + " "\
        + str( wavelength[0] ) + "-" + str( wavelength[-1] ) + " nm."

    outdata = create_raster_like(hyp_img, output_filename, Nlayers=num_bands, outtype=4, interleave='bip', force=True, 
                            description=description, metadata_keys_copy=['band names', 'wavelength' ], metadata_add=s_f)

    outdata_LR = outdata.open_memmap(writable=True)
    
    #  Reshaping input and output files to 2D  #
    
    input_image_linear = input_image.reshape(num_idx, num_bands)
    inversion_result_linear = inversion_result.reshape(num_idx, num_inv_bands)
    
    outdata_LR = outdata_LR.reshape(num_idx, num_bands)
      
    
    #  Computing corrected LR  #
    
    print()
    print("Please wait! computing corrected leaf reflectance ....")
    
    start = time()    
    if chunk_size == None:
        chunk_size = np.round( 0.5e9 / 128).astype(int)
        if chunk_size > num_idx:
            chunk_size = num_idx
    
    chunk = np.arange(0, num_idx, chunk_size)
    
    for i in range (len(chunk)):
        
        pixel = input_image_linear[chunk[i]:chunk[i]+chunk_size, :]
        pixel = pixel.astype('float')
        pixel /= scale_factor
        
        rhos = inversion_result_linear[ chunk[i]:chunk[i]+chunk_size, 0] 
        ps =  inversion_result_linear[ chunk[i]:chunk[i]+chunk_size, 1]        
        cs = inversion_result_linear[ chunk[i]:chunk[i]+chunk_size, 2]        
        
        corrected_LR = (pixel - cs[:, None]) / (rhos[:, None] + ps[:, None] * pixel)
        outdata_LR[chunk[i]:chunk[i]+chunk_size, :] = corrected_LR

    outdata_LR = outdata_LR.reshape(num_rows, num_cols, num_bands)
    outdata_LR.flush() # writes and saves in the disk
    
    print()
    print(f"{functionname}: process completed!\nProcess completion time = {(time() - start)/60:.2f} mins.")
    
    return 0

def compute_correlation(array1, array2, method=1):
    """
    Computes pearson correlation coefficient between two arrays of equal dimension.
    
    Args:
        array1: ndarray
        array2: ndarray
        method: integer value 1 or 2 (Default = 1)
                1 = remove NaN value from an array and corresponding value at the same index position from another array
                2 = replace NaN values with zeros 

    Result:
        returns pearson correlation coefficient, array1 and array2 if the computation is successful else `False`
        arrays returned might be modified based on the selected method
    """
    
    start = process_time()
    function_name = "compute_correlation()"
    
    # Check the dimension of arrays 
    if not array1.shape == array2.shape:
        print(f"{function_name} Error: Dimensions mismatch between the arrays !")
        return False

    if not (method == int(1) or method == int(2)):
        print(f"{function_name} Error: invalid argument for method\nvalid arguments = 1 or 2 !")
        return False
    
    # Method 1: Check for NaN values and remove it from both arrays
    if method == int(1):     
        mask = ~np.isnan(array1) & ~np.isnan(array2)
        array1 = array1[mask]
        array2 = array2[mask]

    # Method 2: Check for NaN values and replace with zero
    if method == int(2): 
        array1 = np.where(np.isnan(array1), 0, array1)
        array2 = np.where(np.isnan(array2), 0, array2)

    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(array1.flatten(), array2.flatten())
    pearson_correlation = round(correlation_matrix[0, 1], 2)

    print()
    print(f"{function_name}:  Pearson correlation coefficient = {pearson_correlation}.\nComputation time = {(process_time()-start): .2f} secs.")

    return pearson_correlation, array1, array2