from os import path, remove
from spectralinvariant.chunk_processing import chunk_processing_p, chunk_processing_pC, create_chlorophyll_map, compute_inversion_invariants, compute_illumination_corrected_leaf_spectra
# very simple test (and example) script for the functions in chunk_processing.py

def test_chunk_processing_p():
    print("Testing chunk_processing_p()")
    assert chunk_processing_p( hypfile_full, output_filename=output_p  ) == 0, "chunk_proecssing_p(): Execution failed!"
    assert path.exists( output_p + str( ".hdr" ) ) == True, f"chunk_processing_p(): '{ output_p + str('.hdr') }' header file not found!" # check output header file in the disk
    assert path.exists( output_p ) == True, f"chunk_processing_p(): '{ output_p }' data file not found!" # check output data file in the disk

def test_chunk_processing_pC():
    print("Testing chunk_processing_pC()")
    assert chunk_processing_pC( hypfile_full, output_filename=output_pC ) == 0, "chunk_proecssing_pC(): Execution failed!"
    assert path.exists( output_pC + str( ".hdr" ) ) == True, f"chunk_processing_p(): '{ output_pC + str('.hdr')}' header file not found!" 
    assert path.exists( output_pC ) == True, f"chunk_processing_p(): '{ output_pC }' data file not found!" 

def test_create_chlorophyll_map():
    print("Testing create_chlorophyll_map()")
    assert create_chlorophyll_map( hypfile_full, output_filename=output_chl_map  ) == 0, "create_chlorophyll_map(): Execution failed!"
    assert path.exists( output_chl_map + str( ".hdr" ) ) == True, f"chunk_processing_p(): '{ output_chl_map + str('.hdr')}' header file not found!"
    assert path.exists( output_chl_map ) == True, f"chunk_processing_p(): '{ output_chl_map }' data file not found!"

def test_compute_inversion():
    print("Testing compute_inversion_invariants()")
    assert compute_inversion_invariants( hypfile_full, cmap_filename=output_chl_map+str('.hdr'), output_filename=output_inv_invariants ) == 0, "compute_inversion_invariants(): Execution failed!"
    assert path.exists( output_inv_invariants + str(".hdr") ) == True, f"chunk_processing_p(): '{ output_inv_invariants + str('.hdr')}' header file not found!" 
    assert path.exists( output_inv_invariants ) == True, f"chunk_processing_p(): '{ output_inv_invariants }' data file not found!"

def test_illumination_correction():
    print("Testing compute_illumination_corrected_leaf_spectra()")
    assert compute_illumination_corrected_leaf_spectra( hypfile_full, inversion_filename=output_inv_invariants + str( '.hdr' ), output_filename=output_illum_correction ) == 0, "compute_illumination_corrected_leaf_spectra(): Execution failed!"
    assert path.exists( output_illum_correction + str(".hdr") ) == True, f"chunk_processing_p(): '{ output_illum_correction + str('.hdr')}' header file not found!" 
    assert path.exists( output_illum_correction ) == True, f"chunk_processing_p(): '{ output_illum_correction }' data file not found!"


if __name__ == "__main__":

    # hypfile_name = "TAIGA_subset.hdr" # a small airborne hyperspectral image
    hypfile_name = "Julich_20180626_subset02.hdr"
    # hypfile_path = "../data" # relative to the location of this script, part of spectralinvariants
    hypfile_path = r"C:\Users\BKBIJAY\OneDrive - Teknologian Tutkimuskeskus VTT\Documents\data\julich"
    hypfile_full = path.join( hypfile_path, hypfile_name )

    # output file names, (default values used for the test purpose. Can be changed to suitable file names)
    output_p = "Julich_20180626_subset02_p_data"
    output_pC = "Julich_20180626_subset02_pC_data"
    output_chl_map = "Julich_20180626_subset02_chl_content_map"
    output_inv_invariants = "Julich_20180626_subset02_chl_inversion_invariants"
    output_illum_correction = "Julich_20180626_subset02_illumination_corrected"

    functions = [ test_chunk_processing_p, test_chunk_processing_pC, test_create_chlorophyll_map, test_compute_inversion, test_illumination_correction ]
    n = 75
    for i in range( len( functions) ):
        functions[i]()
        print( "-" * n )
    
    print( "All functions passed the test" )
    print( '-' * n)
    
    ### delete output files after the test run ?
    
    ans = str(input( "Do you want to remove test ouptput files? : (Y/N) " ) )

    if ans.upper() == "Y":

        output_files = [ output_p, output_pC, output_chl_map, output_inv_invariants, output_illum_correction ]

        for i in range( len( output_files ) ):
            remove( output_files[i] )
            remove( output_files[i] + str( '.hdr' ) )

        print()
        print( "Test output files removed from the disk!" )
        print()
    
    else:        
        print()
        print("Test output files were not deleted")
        print()


