"""
This example shows how to use the spectral invariant theory to retrieve the true leaf reflectance from a Top-of-Canopy (TOC) reflectance hyperspectral image using the method from Ihalainen et al (2023).

Note that unlike in the paper, we do not apply a sunlit fraction mask in this example.
"""

from re import findall
from pathlib import Path
from os import cpu_count
from time import process_time, time
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from spectral import envi
from spectralinvariant.inversion import PROSPECT_D, pc_fast, minimize_cab, golden_cab

# The PROSPECT model and the inversion function will sometimes encounter division by zero or negative square roots. Let's ignore these warnings for now, although it is not a good idea to do so in general.
np.seterr(all="ignore") 

def get_prospect_params(fname):
    """Reads the PROSPECT parameters from the spectra.txt file
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

    values = findall(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', lines[1])
    values = np.array(values).astype('float')
    N, Cab, Cw, Cm, Car, Cbrown, Anth, Cp, Ccl = values
    return N, Cab, Cw, Cm, Car, Cbrown, Anth, Cp, Ccl


def find_nearest(array, value):
    """Finds the index of array element closest to a given value
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def load_hypdata(fpath, fname):
    """Loads a hyperspectral image from an ENVI file
    """
    img = envi.open(fpath / fname)
    wls = np.array(img.metadata["wavelength"], dtype='float')
    image = img.open_memmap()
    return wls, image / 10000.


def main():
    """Applies illumination condition correction to a (synthetic) hyperspectral image
    """
    # Load the synthetic data
    fname = f'homog_leaves_prospectD74_nosoil_dirglob_vza0_sza30_saa0'
    fpath = Path(f'../data/{fname}')
    wls, image = load_hypdata(fpath, fname+'_T.hdr')

    # Select the red green and blue channels for plotting RGB images
    red = find_nearest(wls, 600)
    green = find_nearest(wls, 550)
    blue = find_nearest(wls, 480)

    # Select wavelengths between 670 and 720 nm for the inversion
    band_a = find_nearest(wls, 670)
    band_b = find_nearest(wls, 720)
    wls_subset = wls[band_a:band_b+1]
    image_subset = image[:,:,band_a:band_b+1]
    nrows, ncols, nbands_subset = image_subset.shape
    npix = nrows*ncols

    # Get the true leaf spectral reflectance used in creating the synthetic data
    spectra = np.loadtxt(fpath / 'spectra.txt')
    reflectance = spectra[:,1] # Use index 0 for wavelengths, 2 for leaf transmittance, 3 for forest floor reflectance, and 4 for direct-to-global irradiance ratio

    # The following PROSPECT parameters were used in creating the synthetic data. These are not really needed in this example but they're nice to have
    N, Cab, Cw, Cm, Car, Cbrown, Anth, Cp, Ccl = get_prospect_params(fpath / 'spectra.txt')

    # Create an instance of the PROSPECT class with the input values specified specified by Ihalainen et al. (2023)
    # d = PROSPECT_D(N=1.5, Car=1.0, Cw=0.0, Cm=0.0)
    d = PROSPECT_D(N=N, Car=1.0, Cw=Cw, Cm=Cm) # When using a priori values for N, Cw, and Cm, the inversion naturally becomes even more accurate
    # Set the wavelengths used in the inversion
    d.subset(wls_subset)

    # Reshape the hyperspectral image to make parallel computing easier
    pixels_subset = (image_subset).reshape((npix, nbands_subset))

    # List of the available optimization methods for `minimize_cab()`. SLSQP is the fastest, but not quite as accurate as, e.g., Nelder-Mead or Powell.
    methods = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr']
    
    # Here's how you use the inversion functions `minimize_cab` for a single pixel using
    pixel = image_subset[60,60]
    # Inversion using scipy.minimize (Includes various methods)
    for method in methods:
        ans = minimize_cab(d, pixel, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, initial_guess=30., bounds=[(1., 100.)], method=method)
        print(f'Inversion result for a single pixel using {method}. Cab = {ans:.2f}.\tThe true value is {Cab:.2f}')
    # Inversion using scipy.golden (Golden section search method). This is faster than most methods included in scipy.minimize, but it could be made even faster by writing, e.g., a C extension for the method.
    ans = golden_cab(d, pixel, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, bounds=(1., 100.))
    print(f'Inversion result for a single pixel using golden section search. Cab = {ans:.2f}.\tThe true value is {Cab:.2f}')

    # Run the inversion for the whole image on all CPUs using Parallel to make reduce the computation time. If you're not familiar with Parallel, see https://joblib.readthedocs.io/en/latest/parallel.html for details
    print(f'Starting the inversion on {npix} pixels. This might take a while.')

    start = time()
    # inversion_results = Parallel(n_jobs=cpu_count())(delayed(minimize_cab)(
        # d, pixel, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, initial_guess=[30.], bounds=[(1., 100.)], method=methods[3]) for pixel in pixels_subset)
    inversion_results = Parallel(n_jobs=cpu_count())(delayed(golden_cab)(
        d, pixel, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, bounds=(1., 100.)) for pixel in pixels_subset)

    stop = time()
    print(f'Finished! The inversion took {(stop - start):.2f} seconds.')


    # Define a numpy array for the results and initialize arrays for the spectral invariant parameters (the inversion output only produces Cab values - the spectral invariants need to be computed separately)
    cabs = np.array(inversion_results)
    rhos = np.zeros(npix, dtype=float)
    ps = np.zeros(npix, dtype=float)
    cs = np.zeros(npix, dtype=float)


    print('Computing the spectral invariants.')
    start = process_time()
    # Might be useful to parallelize this loop for larger images...
    for i, pixel in enumerate(pixels_subset):
        rho, p, c = pc_fast(pixel, d.PROSPECT(Cab=cabs[i]))
        rhos[i] = rho
        ps[i] = p
        cs[i] = c

    stop = process_time()
    print(f'Finished! The computation took {(stop - start):.2f} seconds.')

    # Reshape our arrays for plotting
    cabs = cabs.reshape((nrows, ncols))
    rhos = rhos.reshape((nrows, ncols))
    ps = ps.reshape((nrows, ncols))
    cs = cs.reshape((nrows, ncols))

    # Calculate the inversion image and get the median spectra from the image
    inverted_image = (image - cs[:,:,None]) / (rhos[:,:,None] + ps[:,:,None] * image)
    inverted_median = np.nanmedian(inverted_image, axis=(0,1))
    

    ########################
    ### Plot the results ###
    ########################
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5), tight_layout=True)
    fig.suptitle('Inversion results')
    ax = axes[0,0]
    im = ax.imshow(image[:,:,(red, green, blue)]*4)
    ax.set_title('Original RGB')
    ax.axis('off')

    ax = axes[1,0]
    im = ax.imshow(inverted_image[:,:,(red, green, blue)]*4)
    ax.set_title('Inverted RGB')
    ax.axis('off')

    ax = axes[0,1]
    im = ax.imshow(cabs, cmap='Greens', vmin=1, vmax=100)
    ax.set_title(r'$C_{ab}$')
    cb = fig.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[0,2]
    im = ax.imshow(cs, cmap='Spectral', vmin=0., vmax=.01)
    ax.set_title(r'$c$')
    cb = fig.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1,1]
    im = ax.imshow(rhos,cmap='afmhot', vmin=0, vmax=2)
    ax.set_title(r'$\rho$')
    cb = fig.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1,2]
    im = ax.imshow(ps, cmap='cividis', vmin=0.5, vmax=1.5)
    ax.set_title(r'$p$')
    cb = fig.colorbar(im, ax=ax)
    ax.axis('off')


    fig, axes = plt.subplots(1, 2, figsize=(10,5), tight_layout=True)
    ax = axes[0]
    ax.plot(wls, reflectance, c='k', linewidth=3, label='True leaf reflectance')
    ax.plot(wls, inverted_median, linewidth=3, linestyle='--', alpha=0.75, label='Median inverted reflectance')
    ax.legend(frameon=False)

    ax = axes[1]
    ax.scatter(inverted_median, reflectance)
    ax.plot([0, 0.5], [0,0.5], linestyle='dotted', c='k')

    plt.show()


if __name__ == "__main__":
    main()