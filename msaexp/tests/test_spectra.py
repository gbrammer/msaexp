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
            # [120.2, 2.5],  # With SpectrumSampler fits
            [2.293e+12, 8.078e+10], # With template refactor
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
    for with_numba in [False, True]:
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
            assert np.allclose(
                utils.trapz(gau, spec.spec_wobs), 1.0, rtol=5.0e-2
            )

            gau2 = spec.fast_emission_line(line_um, line_flux=1, **kws)
            assert np.allclose(
                utils.trapz(gau2, spec.spec_wobs), 1.0, rtol=1.0e-3
            )

    igm1 = spec.igm_absorption(1.0)
    assert np.allclose(igm1, 1.0)

    igm7 = spec.igm_absorption(7.0)
    assert (igm7[40] < 1.0) & (igm7[40] > 0)

    igm10 = spec.igm_absorption(10.0)
    n10 = igm10 < 1
    assert n10.sum() == 82


def dummy_spectrum(grating="PRISM", filter="CLEAR"):
    """
    Generate a dummy spectrum object
    """
    import grizli.utils

    wave = utils.get_standard_wavelength_grid(
        grating,
        sample=1.05,
    )

    tab = grizli.utils.GTable()
    tab["wave"] = wave
    tab["flux"] = wave * 0. + 1
    tab["err"] = wave * 0. + 0.01
    tab.meta["GRATING"] = grating
    tab.meta["FILTER"] = filter

    return spectrum.SpectrumSampler(tab)


def test_all_gratings():
    """
    Test initializing spectra for all NIRSpec gratings
    """

    from eazy.utils import CLIGHT
    import eazy.templates

    has_orders = {
        ('F070LP', 'G140M'): True,
        ('F100LP', 'G140M'): True,
        ('F170LP', 'G235M'): True,
        ('F290LP', 'G395M'): False,
        ('F070LP', 'G140H'): True,
        ('F100LP', 'G140H'): False,
        ('F170LP', 'G235H'): False,
        ('F290LP', 'G395H'): False,
        ('CLEAR', 'PRISM'): False
    }

    twave = np.arange(0.4, 6, 0.0001)
    tflux = CLIGHT * 1.e10 / (twave * 1.e4)**2

    templ = eazy.templates.Template(arrays=(twave * 1.e4, tflux))

    max_with_orders = {
        ('F070LP', 'G140M'): 1.764,
        ('F100LP', 'G140M'): 1.250,
        ('F170LP', 'G235M'): 1.409,
        ('F070LP', 'G140H'): 1.828,
    }

    orders_line_wave = {
        ('F070LP', 'G140M'): 0.8,
        ('F100LP', 'G140M'): 1.0,
        ('F170LP', 'G235M'): 1.7,
        ('F070LP', 'G140H'): 0.9,
    }

    for optics in has_orders:
        (filter, grating) = optics
        spec = dummy_spectrum(grating=grating, filter=filter)
        res = spec.resample_eazy_template(templ, orders=[1])
        np.allclose(res, 1., atol=1.e-4)

        assert((spec.sensitivity[2] is not None) == has_orders[optics])

        if has_orders[optics]:
            res2 = spec.resample_eazy_template(templ)

            forder = []
            lorder = []

            for m in spec.sensitivity:
                if spec.sensitivity[m] is not None:
                    forder.append(spec.sensitivity[m] / spec.sensitivity[1])

                    ###
                    # Emision line: max intensity should be the ratio
                    # of the order sensitivity to order=1 since the resolution
                    # scales by m and the wavelength where the line appears
                    # also scales by m.
                    #
                    # scale_disp=0.1 for unresolved line to reduce pixel
                    # effects in the test
                    sline = spec.fast_emission_line(
                        orders_line_wave[optics], scale_disp=0.1,
                        orders=[m]
                    )
                    sens_line = np.interp(
                        orders_line_wave[optics] * m, spec["wave"], forder[-1],
                    )
                    lorder.append(sline / sens_line)

            line_max = np.nanmax(lorder, axis=1)
            # print(optics, line_max / line_max[0])

            assert np.allclose(line_max[0], line_max[1:], rtol=0.05)

            forder_tot = np.array(forder).sum(axis=0)
            is_valid = np.isfinite(forder_tot)

            assert is_valid.sum() > 100
            assert np.allclose(res2[is_valid], forder_tot[is_valid], atol=1.e-3)

            assert np.allclose(
                max_with_orders[optics], np.nanmax(res2),
                atol=1.e-2
            )
