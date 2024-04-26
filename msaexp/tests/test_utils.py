from .. import utils


def test_import():
    """ """
    import msaexp.pipeline
    import msaexp.spectrum
    import msaexp.utils
    import msaexp.slit_combine
    import msaexp.drizzle


def test_wavelength_grids():

    import grizli.utils

    grizli.utils.set_warnings()

    for gr in utils.GRATING_LIMITS:
        grid = utils.get_standard_wavelength_grid(gr)
        grid = utils.get_standard_wavelength_grid(gr, log_step=True)

    grid = utils.get_standard_wavelength_grid("prism", free_prism=False)


def test_fwhm():

    import numpy as np

    wave = np.linspace(0.8, 5.0, 128)
    psf_fwhm = utils.get_nirspec_psf_fwhm(wave)

    assert psf_fwhm.min() > 0.2
    assert psf_fwhm.max() < 7
