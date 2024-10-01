# spectralinvariant

Hyperspectral image analysis tools for Python using the spectral invariant theory. The repository also includes an implementations of several versions of the PROSPECT leaf radiative transfer model.

# Installation (local)
You can install the package on your system locally as follows

- Clone this repo to your machine via GitHub Desktop or using git:

```shell
$ git clone https://github.com/mottus/spectralinvariant

```

- Install the package in your repo directory using pip:

```shell
/spectralinvariant$ pip install -e .
```

# Examples
The `examples` directory contains scripts for running the spectral invariant algorithms. They include examples for performing illumination correction using methods appearing in journal articles.
- `random_forest_example_RSE_D_24_02041.py` - Random Forest Regression based method by [Ihalainen et al. (2024)](http://dx.doi.org/10.2139/ssrn.4903267).
- `inversion_example_RSE_D_23_00736.py` - Regularized inversion method using PROSPECT by [Ihalainen et al. (2023)](https://doi.org/10.1016/j.rse.2023.113810).
- `processing_example.py` - Running spectral invariant algorithms on large hyperspectral images.

# References
- Ihalainen, O., Juola, J., & Mõttus, M. (2023). Physically based illumination correction for sub-centimeter spatial resolution hyperspectral data. _Remote Sensing of Environment_, _298_, 113810, https://doi.org/10.1016/j.rse.2023.113810
- Ihalainen, O., Sandmann, T., Rascher, U., & Mõttus, M. (2024). Illumination correction for close-range hyperspectral images using spectral invariants and random forest regression. *Manuscript submitted for publication*, http://dx.doi.org/10.2139/ssrn.4903267
