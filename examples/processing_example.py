from os import path
from spectralinvariant.chunk_processing import chunk_processing_p, chunk_processing_pC, create_chlorophyll_map, compute_inversion_invariants, compute_illumination_corrected_leaf_spectra
# very simple test (and example) script for the functions in chunk_processing.py

hypfile_name = "TAIGA_subset.hdr" # a small airborne hyperspectral image
hypfile_path = "../data" # relative to the location of this script, part of spectralinvariants

hypfile_full = path.join( hypfile_path, hypfile_name )

def test_chunk_processing_p():
    print("Testing chunk_proecssing_p()")
    assert chunk_processing_p( hypfile_full ) == 0, "chunk_proecssing_p(): Execution failed!"

def test_chunk_processing_pC():
    print("Testing chunk_proecssing_pC()")
    assert chunk_processing_pC( hypfile_full ) == 0, "chunk_proecssing_pC(): Execution failed!"

def test_create_chlorophyll_map():
    print("Testing create_chlorophyll_map()")
    assert create_chlorophyll_map( hypfile_full ) == 0, "create_chlorophyll_map(): Execution failed!"

def test_compute_inversion():
    print("Testing compute_inversion_invariants()")
    assert compute_inversion_invariants( hypfile_full ) == 0, "compute_inversion_invariants(): Execution failed!"

def test_illumination_correction():
    print("Testing compute_illumination_corrected_leaf_spectra()")
    assert compute_illumination_corrected_leaf_spectra( hypfile_full ) == 0, "compute_illumination_corrected_leaf_spectra(): Execution failed!"


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
    print("All functions passed the test")


