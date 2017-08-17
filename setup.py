#!/usr/bin/env python
# -*- coding: utf-8 -*-

# LVM Zoo, some Latent Variable Models using GPflow
# Copyright (C) 2017  Nicolas Knudde, Joachim van der Herten

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
import re

VERSIONFILE="lvmzoo/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='LVM Zoo',
      version=verstr,
      author="Nicolas Knudde, Joachim van der Herten",
      author_email="joachim.vanderherten@ugent.be",
      description=("Some Latent Variable Models using GPflow"),
      license="GNU General Public License v3",
      keywords="GP-LVM",
      url="http://github.com/javdrher/lvm-zoo",
      package_data={},
      include_package_data=True,
      ext_modules=[],
      packages=["lvmzoo"],
      package_dir={'lvmzoo': 'lvmzoo'},
      py_modules=['lvmzoo.__init__'],
      test_suite='testing',
      install_requires=['numpy>=1.9', 'scipy>=0.16', 'GPflow>=0.3.5'],
      extras_require={'tensorflow': ['tensorflow>=1.0.0'],
                      'tensorflow with gpu': ['tensorflow-gpu>=1.0.0'],
                      'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc', 'nbsphinx', 'jupyter'],
                      'test': ['nose', 'coverage', 'six', 'parameterized', 'nbconvert', 'nbformat','jupyter',
                               'jupyter_client', 'matplotlib']
                      },
      dependency_links=['https://github.com/GPflow/GPflow/tarball/master#egg=GPflow-0.3.5'],
      classifiers=['License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Natural Language :: English',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
