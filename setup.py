#!/usr/bin/env python

import re
import sys
import asterclient
from setuptools import setup

def parse_dependency_links(file_name):
    dependency_links = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'\s*-[ef]\s+', line):
            dependency_links.append(re.sub(r'\s*-[ef]\s+', '', line))

    return dependency_links


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
      dependency_links=parse_dependency_links('requirements.txt'),
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
