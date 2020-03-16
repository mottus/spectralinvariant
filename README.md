# spectralinvariant

(hyper)spectral data processing with spectral invariants (p-theory)

## Installation
I haven't yet made an official pip package of `spectralinvariant`. But here's how you can install the package on your system locally:

- Clone this repo to your machine via GitHub Desktop or using git:

```shell
$ git clone https://github.com/mottus/spectralinvariant

```

- Install the package in your repo directory using pip:

```shell
/spectralinvariant$ pip install -e .
```

The `-e` flag makes pip install the package locally, i.e., in the `src` directory, __NOT__ within your Python distribution.

## Testing

You can test whether the installation succeeded by running python and importing the `prospect_wavelengths()` function:
```python
>>> import spectralinvariant.prospect
>>> spectralinvariant.prospect.prospect_wavelengths()
```
This should yield
```python
array([ 400,  401,  402, ..., 2498, 2499, 2500])
```
You can also running the `GUI_p` or `hyperspectral_demo.py` which are located in the `demos` folder.