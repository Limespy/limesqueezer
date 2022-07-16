#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from src.limesqueezer.__init__ import __version__ as version

import pandoc

import os
import pathlib
import re
from setuptools import find_packages
from setuptools import setup

path_package = pathlib.Path(__file__).parent

source = 'src'
Python_version = '>=3.10'
changelog_fname = 'CHANGELOG.rst'
license_fname = 'LICENSE.txt'
path_src = path_package / source

def read(*names):
    with open(path_package.joinpath(*names), 'r', encoding = 'utf8') as f:
        return f.read()

# Licence name
with open(path_package / license_fname, 'r', encoding = 'utf8') as f:
    license_name = f.readline()[:-1]

name = tuple(path_src.rglob('__init__.py'))[0].parent.stem

# Long description
# Converting the markdown format to rst
readme_md = read('README.md')
readme_rst = pandoc.write(pandoc.read(readme_md), format = 'rst')
with open(path_package / 'README.rst', 'w', encoding = 'utf8', newline = '') as f:
    f.write(readme_rst)
badges = read('badges.rst')
changelog = read(changelog_fname)
long_description = f'{badges}\n\n{readme_rst}\n{changelog}'



# For verification
print(f'{"-"*60}\n\tSoftware description\n{"-"*60}\n\n{long_description}\n\n') 

github_URL = f'https://github.com/limespy/{name}'


# Loading the list of dependencies
with open(path_package / 'dependencies.txt', encoding = 'utf8') as f:
    dependencies = [line.rstrip() for line in f.readlines()]

def c(*args):
    out = f'{args[0]} :: {args[1]}'
    for arg in args[2:]:
        out += f' :: {arg}'
    return out

def cset(key, *values):
    out = []
    if isinstance(key, str):
        key = (key, )
    for value in values:
        if isinstance(value, tuple):
            out.append(c(*key, *value))
        else:
            out.append(c(*key, value))
    return out

setup(name                 = name,
      version              = version,
      license              = f'{license_name.split()[0]}',
      description          = 'Lossy compression for smooth numerical data series',
      long_description     = long_description,
      author               = 'Limespy',
      author_email         = '',
      url                  = github_URL,
      packages             = find_packages(source),
      package_dir          = {'': source},
      py_modules           = [path.stem for path in path_src.rglob('*.py')],
      include_package_data = True,
      zip_safe             = False,
      classifiers = [
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
                     c('Development Status', '3 - Alpha'),
                     *cset('Intended Audience',
                           'Developers', 'Science/Research'),
                     c('License', 'OSI Approved', license_name),
                     *cset('Operating System',
                           'Unix', 'POSIX', ('Microsoft', 'Windows')),
                     *cset(('Programming Language', 'Python'),
                           '3', ('3', 'Only'), Python_version[2:]),
                     c('Topic', 'Scientific/Engineering'),
                     *cset(('Topic', 'Scientific/Engineering'),
                           'Chemistry', 'Physics'),
                     *cset('Topic',
                           ('System', 'Archiving', 'Compression'), 'Utilities'),
    ],
      project_urls = {'Changelog': f'{github_URL}/blob/master/{changelog_fname}',
                      'Issue Tracker': f'{github_URL}/issues',
    },
      keywords = [
    ],
      python_requires = Python_version,
      install_requires = dependencies,
      extras_require = {
    },
      entry_points = {'console_scripts': [f'{name} = {name}.CLI:main',]
                      },
)
