"""
Fix units of the dispersion tables to be astropy compatible
"""

import astropy.units as u
from astropy.table import Table

import glob

files = glob.glob("*disp.fits")
files.sort()

for file in files:
    tab = Table.read(file)
    tab["WAVELENGTH"].unit = u.micron
    tab["DLDS"].unit = u.micron / u.pixel
    tab["R"].unit = None
    tab.write(file, overwrite=True)
