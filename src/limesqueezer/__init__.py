#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Limesqueezer
========================================================================

Lossy compression tools for smooth data series.

Recommended abbreviation is 'ls'

import limesqueezer as ls
'''

__version__ = '1.0.10'
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
__all__ = ['API']
from .API import *
# A part of the hack to make the package callable
sys.modules[__name__].__class__ = Pseudomodule