========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - |
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |version| image:: https://img.shields.io/pypi/v/limesqueezer.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/limesqueezer

.. |wheel| image:: https://img.shields.io/pypi/wheel/limesqueezer.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/limesqueezer

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/limesqueezer.svg
    :alt: Supported versions
    :target: https://pypi.org/project/limesqueezer

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/limesqueezer.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/limesqueezer

.. |commits-since| image:: https://img.shields.io/github/commits-since/limespy/limesqueezer/v1.0.6.svg
    :alt: Commits since latest release
    :target: https://github.com/limespy/limesqueezer/compare/v1.0.6...master



.. end-badges

Lossy compression tools for smooth data series. WIP

* Free software: MIT license

Installation
============

::

    pip install limesqueezer

You can also install the in-development version with::

    pip install https://github.com/limespy/limesqueezer/archive/master.zip


Documentation
=============


To use the project:

.. code-block:: python

    import limesqueezer
    limesqueezer.longest()


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
