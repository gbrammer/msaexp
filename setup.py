#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools.extension import Extension

import subprocess

import os

#update version
args = 'git describe --tags'
p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
version = p.communicate()[0].decode("utf-8").strip()

#### Versions
# version = "0.1" # init
#version = "0.2" # with trivial tests
version = "0.3" # with 2d plotting

# Set this to true to add install_requires to setup
if True:
    install_requires=[
         'numpy>=1.23.0',
         'cython',
         'matplotlib>=3.5',
         'scipy>=1.9',
         'tqdm',
         'astropy>=5.0',
         'pysiaf',
         'jwst>=1.8',
         'astroquery',
         'grizli',
         'mastquery'
         ]
         
else:
    install_requires = []    
    
#lines = open('grizli/version.py').readlines()
version_str = """# git describe --tags
__version__ = "{0}"\n""".format(version)
fp = open('msaexp/version.py','w')
fp.write(version_str)
fp.close()
print('Git version: {0}'.format(version))

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "msaexp",
    version = version,
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "msaexp: Manual extraction of JWST NIRSpec MSA spectra directly from the telescope exposures",
    license = "MIT",
    url = "https://github.com/gbrammer/msaexp",
    download_url = "https://github.com/gbrammer/msaexp/tarball/{0}".format(version),
    packages=['msaexp'],
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=install_requires,
    package_data={'msaexp': ['data/*fits']}, 
)
