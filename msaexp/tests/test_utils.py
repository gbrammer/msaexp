from .. import utils

def test_import():
    """
    """
    import msaexp.pipeline
    import msaexp.spectrum
    import msaexp.utils


def test_wavelength_grids():
    
    import grizli.utils
    grizli.utils.set_warnings()
    
    for gr in utils.GRATING_LIMITS:
        grid = utils.get_standard_wavelength_grid(gr)
        grid = utils.get_standard_wavelength_grid(gr, log_step=True)
        
    grid = utils.get_standard_wavelength_grid('prism', free_prism=False)


def test_meta_parser():
    
    uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
    regs = utils.regions_from_metafile(uri+'jw02756001001_01_msa.fits',
                                       as_string=True,
                                       with_bars=True)
    