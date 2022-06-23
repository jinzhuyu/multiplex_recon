#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:22:27 2021

@author: jayz
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("multi_net_cython.pyx")
)