#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#%%═════════════════════════════════════════════════════════════════════
# IMPORT


import pathlib
from setuptools import find_packages
from setuptools import setup
import sys

if '--print' in sys.argv:
    import colorama as col
    RESET = col.Style.RESET_ALL
    BLACK = col.Fore.BLACK
    BLUE = col.Fore.BLUE
    CYAN = col.Fore.CYAN
    GREEN = col.Fore.GREEN
    MAGENTA = col.Fore.MAGENTA
    RED = col.Fore.RED
    YELLOW = col.Fore.YELLOW
    WHITE = col.Fore.WHITE
    WHITE_BG = col.Back.WHITE
#%%═════════════════════════════════════════════════════════════════════
# SETUP GLOBALS
BASE_DIR = pathlib.Path(__file__).parent
SOURCE_NAME = 'src'
PYTHON_VERSION = '>=3.10'
PATH_LICENCE = tuple(BASE_DIR.glob('LICENSE*'))[0]
PATH_SCR = BASE_DIR / SOURCE_NAME
PATH_INIT = next(PATH_SCR.rglob('__init__.py'))
PATH_README = tuple(BASE_DIR.glob('README*'))[0]
#%%═════════════════════════════════════════════════════════════════════
# Run tests first
if '--tests' in sys.argv:
    import tests
    print('Running typing checks')
    typing_test_result = tests.typing(shell = False)
    failed = not typing_test_result[0].startswith('Success')
    failed |= bool(typing_test_result[1])
    print(f'{RED if failed else GREEN}{typing_test_result[0]}{RESET}')
    if typing_test_result[1]:
        print(typing_test_result[1])

    print('Running unit tests')
    unit_test_result = tests.unittests(verbosity = 1)
    failed |= bool(unit_test_result.errors)
    failed |= bool(unit_test_result.failures)
    if failed:
        raise Exception('Tests did not pass, read above')
    sys.argv.pop(sys.argv.index('--tests'))
#%%═════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS
def header(text: str, linechar = '─', endchar = '┐', headerwidth  =  60):
    titlewidth = headerwidth // 2
    textlen = len(text)
    l_len = ((titlewidth - textlen) // 2 - 1)
    lpad = linechar*l_len
    rpad = f'{(headerwidth - l_len - textlen - 3)*linechar}'
    return f'{lpad} {GREEN}{text}{RESET} {rpad}{endchar}'
#───────────────────────────────────────────────────────────────────────
# For classifiers
def c(*args):
    out = f'{args[0]} :: {args[1]}'
    for arg in args[2:]:
        out += f' :: {arg}'
    return out
#───────────────────────────────────────────────────────────────────────
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
#%%═════════════════════════════════════════════════════════════════════
# SETUP INFO
if '--print' in sys.argv:
    print(f'\n{header("Starting packaging setup", "=", "=")}\n')
setup_info = {}
# Getting package name
setup_info['name'] = PATH_INIT.parent.stem
#───────────────────────────────────────────────────────────────────────
# Version
with open(PATH_INIT, 'r', encoding = 'utf8') as f:
    while not (line := f.readline().lstrip()).startswith('__version__'):
        pass
    setup_info['version'] = line.split('=')[-1].strip(" '")
#───────────────────────────────────────────────────────────────────────
# Licence
with open(PATH_LICENCE, 'r', encoding = 'utf8') as f:
    LICENSE_NAME = f'{f.readline().strip().split()[0]}'
setup_info['license'] = LICENSE_NAME
#───────────────────────────────────────────────────────────────────────
# Author
setup_info['author'] = 'Limespy'
#───────────────────────────────────────────────────────────────────────
# Author Email
setup_info['author_email'] = ''
#───────────────────────────────────────────────────────────────────────
# URL
setup_info['url'] = f'https://github.com/{setup_info["author"]}/{setup_info["name"]}'
GITHUB_MAIN_URL = f'{setup_info["url"]}/blob/main/'
#───────────────────────────────────────────────────────────────────────
# Description
with open(PATH_README, 'r', encoding = 'utf8') as f:
    # The short description is in the README after badges
    while (description := f.readline().lstrip(' ')).startswith(('#', '\n', '[')):
        pass

    while not (line := f.readline().lstrip(' ')).startswith('\n'):
        description += line

setup_info['description'] = description
#───────────────────────────────────────────────────────────────────────
# Long Description
with open(PATH_README, 'r', encoding = 'utf8') as f:
    setup_info['long_description'] = f.read().replace('./', GITHUB_MAIN_URL)
if PATH_README.suffix == '.md':
    setup_info['long_description_content_type'] = 'text/markdown'
elif PATH_README.suffix != '.rst':
    raise TypeError(f'README file type not recognised: {PATH_README}')
#───────────────────────────────────────────────────────────────────────
# packages
setup_info['packages']  = find_packages(SOURCE_NAME)
#───────────────────────────────────────────────────────────────────────
# Packages Dir
setup_info['package_dir']  = {'': SOURCE_NAME}
#───────────────────────────────────────────────────────────────────────
# Py Modules
setup_info['py_modules'] = [path.stem for path in PATH_SCR.rglob('*.py')]
#───────────────────────────────────────────────────────────────────────
# Include  Package Data
setup_info['include_package_data'] = True
#───────────────────────────────────────────────────────────────────────
# Classifiers
# complete classifier list:
#   http://pypi.python.org/pypi?%3Aaction=list_classifiers
setup_info['classifiers']   = [
    c('Development Status', '3 - Alpha'),
    *cset('Intended Audience', 'Developers', 'Science/Research'),
    c('License', 'OSI Approved', LICENSE_NAME),
    *cset('Operating System', 'Unix', 'POSIX', ('Microsoft', 'Windows')),
    *cset(('Programming Language', 'Python'),
          '3', ('3', 'Only'), PYTHON_VERSION[-4:]),
    c('Topic', 'Scientific/Engineering'),
    *cset(('Topic', 'Scientific/Engineering'), 'Chemistry', 'Physics'),
    *cset('Topic', ('System', 'Archiving', 'Compression'), 'Utilities'),
                               ]
#───────────────────────────────────────────────────────────────────────
# Project URLs
setup_info['project_urls'] = {
    'Changelog': f'{GITHUB_MAIN_URL}{PATH_README.name}#Changelog',
    'Issue Tracker': f'{setup_info["url"]}/issues'}
#───────────────────────────────────────────────────────────────────────
# Keywords
setup_info['keywords'] = ['compression', 'numpy']
#───────────────────────────────────────────────────────────────────────
# Python requires
setup_info['python_requires']  = PYTHON_VERSION
#───────────────────────────────────────────────────────────────────────
# Install requires
with open(BASE_DIR / 'dependencies.txt', encoding = 'utf8') as f:
    setup_info['install_requires'] = [line.rstrip() for line in f.readlines()]
#───────────────────────────────────────────────────────────────────────
# Extras require
with open(BASE_DIR / 'dependencies_dev.txt', encoding = 'utf8') as f:
    setup_info['extras_require'] = {'dev': [line.rstrip() for line in f.readlines()]}
#───────────────────────────────────────────────────────────────────────
# Entry points
setup_info['entry_points'] = {'console_scripts':
[f'{setup_info["name"]} = {setup_info["name"]}.CLI:main',]}
#%%═════════════════════════════════════════════════════════════════════
# PRINTING SETUP INFO
if '--print' in sys.argv:
    for key, value in setup_info.items():
        print(f'\n{header(key)}\n')
        if isinstance(value, list):
            print('[', end = '')
            if value:
                print(value.pop(0), end = '')
                for item in value:
                    print(f',\n {item}', end = '')
            print(']')
        elif isinstance(value, dict):
            print('{', end = '')
            if value:
                items = iter(value.items())
                key2, value2 = next(items)
                print(f'{key2}: {value2}', end = '')
                for key2, value2 in items:
                    print(f',\n {key2}: {value2}', end = '')
            print('}')
        else:
            print(value)
    print(f'\n{header("Calling setup", "=", "=")}\n')
    sys.argv.pop(sys.argv.index('--print'))
#%%═════════════════════════════════════════════════════════════════════
# RUNNING THE SETUP
setup(**setup_info)
