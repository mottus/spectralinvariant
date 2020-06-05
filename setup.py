#!/usr/bin/env python
#-*- encoding: utf-8 -*-
"""
Some good packaging practices:
blog.ionelmc.ro/2014/05/25/python-packaging
"""

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

"""
def read(*names, **kwargs):
	with io.open(
		join(dirname(__file__), *names),
		encoding=kwargs.get('encoding', 'utf8')
		) as fh:
			return fh.read()
"""

with open('README.md', 'r') as f:
	long_description = f.read()


setup(name='spectralinvariant', # Name is it will appear on PyPi
	version='0.0.1',
	description='Spectral invariants package',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
	classifiers=[
	'Programming Language :: Python :: 3',
	'Licence :: GNU General Public Licence v3.0',
	'Operating System :: OS Independent'],
	long_description=long_description,
	long_description_content_type='text/markdown',
	python_requires='>=3.6',
	url='https://github.com/mottus/spectralinvariant',
	author='Author Name',
	author_email='author.name@adress.com'
	)
