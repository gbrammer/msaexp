from .. import msa

def test_meta_parser():
    
    uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
    regs = msa.regions_from_metafile(uri+'jw02756001001_01_msa.fits',
                                       as_string=True,
                                       with_bars=True)
    