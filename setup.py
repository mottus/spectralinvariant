import setuptools
from setuptools import setup

with open('README.md', 'r') as f:
	long_description = f.read()


setup(name='spectralinvariant', # Name is it will appear on PyPi
	version='0.0.1',
	description='Learning how to create a package in Python',
	py_modules=[
	'prospect',
	'spectralinvariants',
	'hypdatatools_algorithms',
	'hypdatatools_img',
	'hypdatatools_utils'],
	package_dir={'': 'src'},
	classifiers=[
	'Programming Language :: Python :: 3',
	'Licence :: GNU General Public Licence v3.0',
	'Operating System :: OS Independent'],
	long_description=long_description,
	long_description_content_type='text/markdown',
	packages=setuptools.find_packages(),
	python_requires='>=3.6',
	url='https://github.com/mottus/spectralinvariant',
	author='Author name',
	author_email='author.name@adress.com'
	)
