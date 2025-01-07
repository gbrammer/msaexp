"""
Test spectrum extractions and fits
"""

import os

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

import astropy.io.fits as pyfits
import jwst

from .. import utils, spectrum

eazy_templates = None

if jwst.__version__ < "100.16":
    # SPECTRUM_FILE = f'ceers-prism.1345_933.v0.spec.fits'
    SPECTRUM_FILE = "test-driz-center-bkg_933.spec.fits"
    # SPECTRUM_FILE = 'test-driz-center_933.spec.fits'
else:
    SPECTRUM_FILE = "rubies-egs51-v4_prism-clear_4233_19489.spec.fits"


def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


def test_load_templates():

    import eazy

    global eazy_templates

    os.chdir(data_path())

    current_path = os.getcwd()

    path = os.path.join(os.path.dirname(eazy.__file__), "data/")
    if not os.path.exists(os.path.join(path, "templates")):
        eazy.fetch_eazy_photoz()

    os.chdir(current_path)
    os.chdir(data_path())

    if not os.path.exists("templates"):
        eazy.symlink_eazy_inputs()

    _param = "templates/sfhz/carnall_sfhz_13.param"
    eazy_templates = eazy.templates.read_templates_file(_param)


def test_fit_redshift():
    """
    Redshift fit with spline + line templates
    """
    global eazy_templates
    global SPECTRUM_FILE

    os.chdir(data_path())

    if not os.path.exists(SPECTRUM_FILE):
        SPECTRUM_FILE = "static-" + SPECTRUM_FILE

    spectrum.FFTSMOOTH = True

    kws = dict(
        eazy_templates=None,
        scale_disp=1.0,
        nspline=33,
        Rline=2000,
        use_full_dispersion=False,
        vel_width=100,
    )

    z = 4.2341
    z0 = [4.1, 4.4]

    fig, spec, zfit = spectrum.plot_spectrum(SPECTRUM_FILE, z=z, **kws)

    fig.savefig(SPECTRUM_FILE.replace("spec.fits", "spec.spl.png"))

    assert "z" in zfit

    assert np.allclose(zfit["z"], z, rtol=0.01)

    assert "coeffs" in zfit
    if "line OIII" in zfit["coeffs"]:
        assert np.allclose(
            zfit["coeffs"]["line OIII"], [2386.17, 35.93], rtol=0.5
        )
    elif "line OIII-5007" in zfit["coeffs"]:
        assert np.allclose(
            zfit["coeffs"]["line OIII-5007"], [1753.03, 35.952727162], rtol=0.5
        )

    if eazy_templates is not None:
        kws["eazy_templates"] = eazy_templates
        kws["use_full_dispersion"] = False
        fig, spec, zfit = spectrum.fit_redshift(
            SPECTRUM_FILE, z0=z0, is_prism=True, **kws
        )

        plt.close("all")
        assert "z" in zfit

        assert np.allclose(zfit["z"], z, rtol=0.01)

        assert "coeffs" in zfit
        assert np.allclose(
            zfit["coeffs"]["4590.fits"],
            # [127.2, 3.418],
            [120.2, 2.5],  # With SpectrumSampler fits
            rtol=0.5,
        )

        #### use_full_dispersion is deprecated now using SpectrumSampler

        # # With dispersion
        # kws['use_full_dispersion'] = True
        # fig, spec, zfit = spectrum.fit_redshift(f'ceers-prism.1345_933.v0.spec.fits',
        #                       z0=z0,
        #                       is_prism=True,
        #                       **kws)
        #
        # plt.close('all')
        # assert('z' in zfit)
        #
        # assert(np.allclose(zfit['z'], z, rtol=0.01))
        #
        # assert('coeffs' in zfit)
        # assert(np.allclose(zfit['coeffs']['4590.fits'],
        #                    [75.95547, 3.7042],
        #                    rtol=0.5))


def test_sampler_object():
    """
    Test the spectrum.SpectrumSampler methods
    """

    os.chdir(data_path())

    spec = spectrum.SpectrumSampler(SPECTRUM_FILE)
    sampler_checks(spec)

    new = spec.redo_1d_extraction()
    sampler_checks(new)

    # Initialized from HDUList
    for with_numba in [True, False]:
        with pyfits.open(SPECTRUM_FILE) as hdul:
            spec = spectrum.SpectrumSampler(hdul)
            sampler_checks(spec, with_numba=with_numba)


def sampler_checks(spec, with_numba=False):

    from ..resample import resample_template as RESAMPLE_FUNC
    from ..resample import sample_gaussian_line as SAMPLE_LINE_FUNC

    if with_numba:
        try:
            from ..resample_numba import (
                resample_template_numba as RESAMPLE_FUNC,
            )
            from ..resample_numba import (
                sample_gaussian_line_numba as SAMPLE_LINE_FUNC,
            )
        except ImportError:
            return True

    spectrum.RESAMPLE_FUNC = RESAMPLE_FUNC
    spectrum.SAMPLE_LINE_FUNC = SAMPLE_LINE_FUNC

    assert np.allclose(spec.valid.sum(), 327, atol=5)

    # emission line
    z = 4.2341
    line_um = 3727.0 * (1 + z) / 1.0e4

    for s in [1, 1.3, 1.8, 2.0]:
        for v in [50, 100, 300, 500, 1000]:
            kws = dict(
                scale_disp=s,
                velocity_sigma=v,
                orders=[1]
            )

            gau = spec.emission_line(line_um, line_flux=1, **kws)
            assert np.allclose(np.trapz(gau, spec.spec_wobs), 1.0, rtol=5.0e-2)

            gau2 = spec.fast_emission_line(line_um, line_flux=1, **kws)
            assert np.allclose(
                np.trapz(gau2, spec.spec_wobs), 1.0, rtol=1.0e-3
            )

    igm1 = spec.igm_absorption(1.0)
    assert np.allclose(igm1, 1.0)

    igm7 = spec.igm_absorption(7.0)
    assert (igm7[40] < 1.0) & (igm7[40] > 0)

    igm10 = spec.igm_absorption(10.0)
    n10 = igm10 < 1
    assert n10.sum() == 82
