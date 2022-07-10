#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
import pathlib
from setuptools import find_packages
from setuptools import setup

from src.limesqueezer.__init__ import __version__ as version

path_package = pathlib.Path(__file__).parent

source = 'src'
Python_version = '>=3.9'
changelog_name = 'CHANGELOG.rst'
license_name = 'LICENSE.txt'

path_src = path_package / source

def read(*names, **kwargs):
    with io.open(path_package.joinpath(*names), 'r',
                 encoding = kwargs.get('encoding', 'utf8')
                ) as fh:
        return fh.read()

badges = re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('badges.rst'))
changelog = re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read(changelog_name))

# Licence name
with io.open(path_package / license_name, 'r') as file:
    license_name = file.readline()[:-1]

name = tuple(path_src.rglob('__init__.py'))[0].parent.stem

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

setup(name = name,
      version = version,
      license = f'{license_name.split()[0]}',
      description = 'Lossy compression tools for smooth data series',
      long_description = f'{badges}\n{changelog}',
      author = 'Limespy',
      author_email = '',
      url = github_URL,
      packages = find_packages(source),
      package_dir = {'': source},
      py_modules = [path.stem for path in path_src.rglob('*.py')],
      include_package_data = True,
      zip_safe = False,
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
      project_urls = {
        'Changelog': f'{github_URL}/blob/master/{changelog_name}',
        'Issue Tracker': f'{github_URL}/issues',
    },
      keywords = [
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
      python_requires = Python_version,
      install_requires = dependencies
      extras_require = {
    },
      entry_points = {
        'console_scripts': [f'{name} = {name}.CLI:main',
        ]
    },
)
