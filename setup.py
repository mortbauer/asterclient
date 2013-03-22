#!/usr/bin/env python

import re
import sys
import asterclient
from setuptools import setup

# from http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
def parse_requirements(file_name):
    requirements = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'(\s*#)|(\s*$)', line):
            continue
        if re.match(r'\s*-e\s+', line):
            requirements.append(re.sub(r'\s*-e\s+.*#egg=(.*)$', r'\1', line))
        elif re.match(r'\s*-f\s+', line):
            pass
        else:
            requirements.append(line)

    return requirements


def parse_dependency_links(file_name):
    dependency_links = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'\s*-[ef]\s+', line):
            dependency_links.append(re.sub(r'\s*-[ef]\s+', '', line))

    return dependency_links


    base = "Win32GUI"


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
      package_data={'asterclient':['data/defaults.yml']},
      install_requires=parse_requirements('requirements.txt'),
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
