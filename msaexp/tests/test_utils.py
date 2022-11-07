
def test_import():
    """
    """
    import msaexp.pipeline
    import msaexp.spectrum
    import msaexp.utils


def test_wavelength_grids():
    
    import msaexp.utils
    import grizli.utils
    grizli.utils.set_warnings()
    
    for gr in msaexp.utils.GRATING_LIMITS:
        grid = msaexp.utils.get_standard_wavelength_grid(gr)
        grid = msaexp.utils.get_standard_wavelength_grid(gr, log_step=True)
        
    grid = msaexp.utils.get_standard_wavelength_grid('prism', free_prism=False)
    