#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python setup.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("multi_net_cython.pyx")
)
