import os
print('Current directory')
print(os.getcwd())

import spectral
import spectral.io.envi as envi

from pathlib import Path
fpath = Path('data')
print(fpath.exists())
# from spectralinvariant import prospect
# print(prospect.prospect_wavelengths())

import numpy as np
print(np.__version__)