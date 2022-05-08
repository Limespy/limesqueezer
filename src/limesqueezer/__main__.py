#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Entrypoint module, in case you use `python -m limesqueezer`.
"""

#%%═════════════════════════════════════════════════════════════════════
# IMPORT
from . import CLI
###═════════════════════════════════════════════════════════════════════
doc = '''
Limesqueezer
========================================================================

Lossy compression tools for smooth data series.

Recommended abbreviation is 'ls'

import limesqueezer as ls
'''
if __name__ == "__main__":
    CLI.main()