#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
import pathlib
from setuptools import find_packages
from setuptools import setup

path_package = pathlib.Path(__file__).parent
source = 'src'
path_src = path_package / source

def read(*names, **kwargs):
    with io.open(path_package.joinpath(*names), 'r',
                 encoding=kwargs.get('encoding', 'utf8')
                ) as fh:
        return fh.read()

badges = re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst'))
changelog = re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))

# Licence name
with io.open(path_package / 'LICENSE.txt', 'r') as file:
    lincensename = file.readline()

setup(name = 'limesqueezer',
      version = '1.0.11',
      license = f'{lincensename.split()[0]}',
      description = 'Lossy compression tools for smooth data series',
      long_description = f'{badges}\n{changelog}',
      author = 'Limespy',
      author_email = '',
      url = 'https://github.com/limespy/limesqueezer',
      packages = find_packages(source),
      package_dir = {'': source},
      py_modules = [path.stem for path in path_src.rglob('*.py')],
      include_package_data = True,
      zip_safe = False,
      classifiers = [
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        f'License :: OSI Approved :: {lincensename}',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: System :: Archiving :: Compression',
        'Topic :: Utilities',
    ],
      project_urls = {
        'Changelog': 'https://github.com/limespy/limesqueezer/blob/master/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/limespy/limesqueezer/issues',
    },
      keywords = [
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
      python_requires = '>=3.10',
      install_requires = [
        'matplotlib ~=3.5.1',
        'numba ~= 0.55.1',
        'numpy ~= 1.21.5'
    ],
      extras_require = {
    },
      entry_points = {
        'console_scripts': [
            'limesqueezer = limesqueezer.CLI:main',
        ]
    },
)
