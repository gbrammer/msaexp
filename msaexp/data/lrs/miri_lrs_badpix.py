import glob
import numpy as np
from tqdm import tqdm

import scipy.ndimage as nd
import astropy.io.fits as pyfits

from grizli import utils
from grizli import nbutils

files = """
Level2/jw08051005001_03103_00001_mirimage_cal.fits
Level2/jw08051005001_03103_00002_mirimage_cal.fits
Level2/jw08051005001_03105_00001_mirimage_cal.fits
Level2/jw08051005001_03105_00002_mirimage_cal.fits
Level2/jw08051006001_03103_00001_mirimage_cal.fits
Level2/jw08051006001_03103_00002_mirimage_cal.fits
Level2/jw08051006001_03105_00001_mirimage_cal.fits
Level2/jw08051006001_03105_00002_mirimage_cal.fits
Level2/jw08051007001_03103_00001_mirimage_cal.fits
Level2/jw08051007001_03103_00002_mirimage_cal.fits
Level2/jw08051007001_03105_00001_mirimage_cal.fits
Level2/jw08051007001_03105_00002_mirimage_cal.fits
Level2/jw08051008001_03103_00001_mirimage_cal.fits
Level2/jw08051008001_03103_00002_mirimage_cal.fits
Level2/jw08051008001_03105_00001_mirimage_cal.fits
Level2/jw08051008001_03105_00002_mirimage_cal.fits
Level2/jw08051009001_03103_00001_mirimage_cal.fits
Level2/jw08051009001_03103_00002_mirimage_cal.fits
Level2/jw08051009001_03105_00001_mirimage_cal.fits
Level2/jw08051009001_03105_00002_mirimage_cal.fits
Level2/jw08051010001_04103_00001_mirimage_cal.fits
Level2/jw08051010001_04103_00002_mirimage_cal.fits
Level2/jw08051010001_04105_00001_mirimage_cal.fits
Level2/jw08051010001_04105_00002_mirimage_cal.fits
Level2/jw08051011001_03103_00001_mirimage_cal.fits
Level2/jw08051011001_03103_00002_mirimage_cal.fits
Level2/jw08051011001_03105_00001_mirimage_cal.fits
Level2/jw08051011001_03105_00002_mirimage_cal.fits
Level2/jw08051012001_03103_00001_mirimage_cal.fits
Level2/jw08051012001_03103_00002_mirimage_cal.fits
Level2/jw08051012001_03105_00001_mirimage_cal.fits
Level2/jw08051012001_03105_00002_mirimage_cal.fits
Level2/jw08051015001_03103_00001_mirimage_cal.fits
Level2/jw08051015001_03103_00002_mirimage_cal.fits
Level2/jw08051015001_03105_00001_mirimage_cal.fits
Level2/jw08051015001_03105_00002_mirimage_cal.fits
Level2/jw08051016001_03103_00001_mirimage_cal.fits
Level2/jw08051016001_03103_00002_mirimage_cal.fits
Level2/jw08051016001_03105_00001_mirimage_cal.fits
Level2/jw08051016001_03105_00002_mirimage_cal.fits
Level2/jw08051017001_03103_00001_mirimage_cal.fits
Level2/jw08051017001_03103_00002_mirimage_cal.fits
Level2/jw08051017001_03105_00001_mirimage_cal.fits
Level2/jw08051017001_03105_00002_mirimage_cal.fits
""".strip().split()

files = glob.glob("Level2/jw08051*cal.fits")
files.sort()

print("\n".join(files))
print(f"\n {len(files)} files")

slx = slice(298, 349)
sly = slice(5, 322)

yp, xp = np.indices((1024, 1032))

data = np.array([
    pyfits.open(file)["SCI"].data[sly, slx] for file in tqdm(files)
])

scl = np.nanmedian(data / data[0], axis=(1,2))
med = np.nanmedian((data.T / scl).T, axis=0)

filter_footprint = utils.make_filter_footprint(
    filter_size=7, filter_central=0,
)[None, :]

filter_med = nd.generic_filter(
    med, nbutils.nanmedian, footprint=filter_footprint.T
)

resid = med - filter_med

hot = resid > 0.2
hot |= resid < -0.2

filter_mean = nd.generic_filter(
    med * np.nan**hot, nbutils.nanmean, footprint=filter_footprint.T
)

resid = med - filter_mean
hot = resid > 0.2
hot |= resid < -0.2

rx = np.nanmean(resid * np.nan**hot, axis=1)

resid = (resid.T - rx).T

hot = resid > 0.2
hot |= resid < -0.2

rfill = resid*1.
rfill[hot | ~np.isfinite(resid)] = 0
nmad = utils.nmad(rfill[~hot & np.isfinite(rfill)])

hot |= (resid < -4 * nmad) | (resid > 4*nmad)
rfill[hot | ~np.isfinite(resid)] = 0

filter_medt = nd.generic_filter(
    med.T, nbutils.nanmedian, footprint=filter_footprint
).T

px = np.array([yp[sly, slx][hot], xp[sly, slx][hot]])

print(f"N={hot.sum()} identified unreliable pixels")

print("write miri_lrs_badpix.txt")

with open("miri_lrs_badpix.txt", "w") as fp:
    fp.write("# yi xi\n")
    for row in px.T:
        fp.write("{0} {1}\n".format(*row))

    
