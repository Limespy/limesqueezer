#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Entrypoint module, in case you use `python -m limesqueezer`.

'''
Limesqueezer
============

Lossy compression tools for smooth data series.

Recommended abbreviation is 'ls'

importing ::

    import limesqueezer as ls

'''

#%%═════════════════════════════════════════════════════════════════════
# IMPORT
from . import CLI
CLI.main()