from os import path, getcwd
from spectralinvariant.chunk_processing import chunk_processing_p, chunk_processing_pC, create_chlorophyll_map, compute_inversion_invariants, compute_illumination_corrected_leaf_spectra

hypfile_name = "test_100_100.hdr" # a small test hypfile

# default file names returned by the function 
cmap_fname = "chl_content_map.hdr"
chl_inversion_fname = "chl_inversion_invariants.hdr"

# change the following file path
hypfile_path = r"c:\\Users\\BKBIJAY\\OneDrive - Teknologian Tutkimuskeskus VTT\\Documents\\data\\Analysis\\test_scrpit" # local file path (and current working directory)

hypfile_full = path.join( hypfile_path, hypfile_name )

def test_chunk_processing_p():
    assert chunk_processing_p( hypfile_full ) == 0, "chunk_proecssing_p() : Execution failed !"

def test_chunk_processing_pC():
    assert chunk_processing_pC( hypfile_full ) == 0, "chunk_proecssing_pC() : Execution failed !"

def test_create_chlorophyll_map():
    assert create_chlorophyll_map( hypfile_full ) == 0, "create_chlorophyll_map() : Execution failed !"

def test_compute_inversion():
    assert compute_inversion_invariants( hypfile_full ) == 0, "compute_inversion_invariants() : Execution failed !"

def test_illumination_correction():
    assert compute_illumination_corrected_leaf_spectra( hypfile_full ) == 0, "compute_illumination_corrected_leaf_spectra() : Execution failed !"


if __name__ == "__main__":
    n = 50
    test_chunk_processing_p()
    print( "-" * n )
    test_chunk_processing_pC()
    print( "-" * n )
    test_create_chlorophyll_map()
    print( "-" * n )
    test_compute_inversion()
    print( "-" * n )
    test_illumination_correction()
    print("All functions passed! the test")
    

    