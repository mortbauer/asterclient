#!/usr/bin/env python

import asterclient
from setuptools import setup

setup(name='asterclient',
      entry_points = {
          'console_scripts' :
              ['asterclient = asterclient.main:main',
               ]},
      version=asterclient.__version__,
      description=asterclient.__description__,
      author=asterclient.__author__,
      author_email=asterclient.__author_email__,
      url=asterclient.__url__,
      download_url=asterclient.__url__,
      license=asterclient.__copyright__,
      packages=['asterclient'],
      package_data={'asterclient':['data/default.conf']},
      extras_require = {
        'autofigure':  ["matplotlib"]
      },
      provides='asterclient',
      classifiers=[
        'Development Status :: Alpha',
        'Topic :: Text Processing :: Markup',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: GNU General Public License (GPL)'],
)
