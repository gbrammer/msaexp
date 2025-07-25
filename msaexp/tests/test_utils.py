import os
import numpy as np
import pytest
from scipy.stats import uniform

import jwst

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


def test_resolution_curves():
    """ """
    for gr in utils.GRATING_LIMITS:

        wgrid = utils.get_standard_wavelength_grid(gr)

        for grating in [gr.upper(), gr.lower()]:
            # calculate grid internally
            R = utils.get_default_resolution_curve(
                grating=grating, wave=None, grating_degree=2
            )

            # With grating fit
            Rg = utils.get_default_resolution_curve(
                grating=grating, wave=wgrid, grating_degree=2
            )

            # Without extrapolation
            Ri = utils.get_default_resolution_curve(
                grating=grating, wave=wgrid, grating_degree=None
            )

            # Only test over first part of the array
            sl = slice(0, 200)
            assert np.allclose(R[sl], Rg[sl], rtol=1.0e-3)
            assert np.allclose(R[sl], Ri[sl], rtol=1.0e-3)

    # LRS prism
    _LRS_R = utils.get_default_resolution_curve(
        grating="LRS", wave=None, grating_degree=2
    )

    # Test extrapolation from fit
    grating = "G235M"
    wgrid = np.array(
        [
            2.0,  # In nominal range
            4.0,  # Extended
        ]
    )

    # With polynomial extrapolation
    for deg in [0, 1, 2, 3]:
        Rg = utils.get_default_resolution_curve(
            grating=grating, wave=wgrid, grating_degree=deg
        )
        # R(2 x lam) ~ 2 * R(lam)
        assert np.allclose(Rg[1] / Rg[0], 2.0, rtol=0.05)


def test_fwhm():

    import numpy as np

    wave = np.linspace(0.8, 5.0, 128)
    psf_fwhm = utils.get_nirspec_psf_fwhm(wave)

    assert psf_fwhm.min() > 0.2
    assert psf_fwhm.max() < 7


def test_bar_correction():

    scaled_yshutter = np.array([0.0, 0.5, 2.0])

    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert not wrapped
    assert bar[0] > 0.95
    assert bar[1] < 0.5
    assert bar[2] < 0.1

    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=True,
        bar_data=None,
    )

    assert not wrapped
    assert bar[0] > 0.95
    assert bar[1] < 0.5
    assert np.isnan(bar[2])

    # num_shutters
    for s in [1, 2, 3]:
        bar, wrapped = utils.get_prism_bar_correction(
            scaled_yshutter,
            num_shutters=s,
            wrap="auto",
            wrap_pad=0.2,
            mask=True,
            bar_data=None,
        )
        assert bar[0] > 0.95
        assert bar[1] < 0.5
        assert np.isnan(bar[2])

    scaled_yshutter = np.array([-3.0, 0.0, 0.5, 2.0, 3.0])
    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert wrapped
    assert bar[0] == bar[1] == bar[3]

    scaled_yshutter = np.array([-3.0, 0.0, 0.5, 2.0, 3.0])
    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap=False,
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert not wrapped
    assert bar[0] == 0

    scaled_yshutter = np.array([0.0, 0.5, 2.0, 3.0])
    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap=True,
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert wrapped
    assert bar[0] == bar[2]

    # Shapes
    scaled_yshutter = np.array([-2.0, 0.0, 0.5, 2.0]).reshape((2, 2))

    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert not wrapped
    assert bar.shape == (2, 2)

    bar_flat, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter.flatten(),
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert np.allclose(bar.flatten(), bar_flat)

    assert np.allclose(
        bar,
        bar_flat.reshape((2, 2)),
    )

    # Wavelength
    scaled_yshutter = np.array([0.0, 0.5, 2.0])
    wavelengths = np.array([1.0, 2.0, 3.0])

    bar, wrapped = utils.get_prism_wave_bar_correction(
        scaled_yshutter,
        wavelengths,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=True,
        bar_data=None,
    )

    assert bar[0] > 0.95
    assert bar[1] < 0.5
    assert np.isnan(bar[2])

    # Bigger 2D array
    sh = (21, 435)

    np.random.seed(1)

    scaled_yshutter = uniform.rvs(loc=-1.5, scale=3, size=sh)
    wavelengths = uniform.rvs(loc=0.8, scale=5.3 - 0.8, size=sh)

    for s in [1, 2, 3]:
        bar, wrapped = utils.get_prism_bar_correction(
            scaled_yshutter,
            num_shutters=3,
            wrap="auto",
            wrap_pad=0.2,
            mask=False,
            bar_data=None,
        )

    assert bar.shape == sh
    assert bar.max() < 1.05
    assert bar.min() < 0.4

    for s in [1, 2, 3]:
        bar, wrapped = utils.get_prism_wave_bar_correction(
            scaled_yshutter,
            wavelengths,
            num_shutters=3,
            wrap="auto",
            wrap_pad=0.2,
            mask=False,
            bar_data=None,
        )

    assert bar.shape == sh
    assert bar.max() < 1.05
    assert bar.min() < 0.4


def test_normalization():

    sh = (21, 435)

    np.random.seed(1)

    wavelengths = uniform.rvs(loc=0.8, scale=5.3 - 0.8, size=sh)

    for q in [1, 2, 3, 4]:
        corr = utils.get_normalization_correction(
            wavelengths,
            q,
            180,
            65,
            grating="PRISM",
            verbose=True,
        )

        assert corr.shape == sh
        assert corr.min() > 0.9
        assert corr.max() < 1.15

    for grating in ["G140M", "G235M", "G395M"]:
        corr = utils.get_normalization_correction(
            wavelengths,
            q,
            180,
            65,
            grating="G235M",
            verbose=True,
        )

        assert np.allclose(corr, 1.0)


def test_badpix():

    utils.cache_badpix_arrays()

    for detector in ["NRS1", "NRS2"]:
        assert detector in utils.MSAEXP_BADPIX

        assert len(utils.MSAEXP_BADPIX[detector]) == 3
        assert utils.MSAEXP_BADPIX[detector][0].shape == (2048, 2048)
        assert "yaml" in utils.MSAEXP_BADPIX[detector][2]


def test_array_bins():

    step = 2
    array = np.arange(0, 11, step)
    bins = utils.array_to_bin_edges(array)

    assert len(bins) == len(array) + 1
    assert np.allclose(bins[:-1], array - step / 2.0)
    assert np.allclose(bins[1:], array + step / 2.0)
    assert np.allclose(np.diff(bins), step)


def test_pixfrac():

    oversample = 8
    pixfrac = 0.5

    steps = utils.pixfrac_steps(oversample, pixfrac)

    assert len(steps) == oversample
    assert steps[0] == -pixfrac + 1.0 / oversample * pixfrac
    assert steps[-1] == pixfrac - 1.0 / oversample * pixfrac

    for oversample in [4, 8, 9, 16]:
        for pixfrac in [0.1, 0.5, 1.0]:
            steps = utils.pixfrac_steps(oversample, pixfrac)

            assert len(steps) == oversample
            assert steps[0] == -pixfrac + 1.0 / oversample * pixfrac
            assert steps[-1] == pixfrac - 1.0 / oversample * pixfrac


def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


def test_slit_things():

    import jwst.datamodels
    import jwst

    os.chdir(data_path())

    if jwst.__version__ < "100.16":
        file = "jw01345062001_03101_00001_nrs2_phot.138.1345_933.fits"
    else:
        file = "jw04233005001_03101_00002_nrs1_phot.140.4233_19489.fits"

    with jwst.datamodels.open(file) as slit:
        utils.update_slit_metadata(slit)

    with jwst.datamodels.open(file) as slit:
        corr = utils.slit_normalization_correction(slit, verbose=True)


@pytest.mark.skipif(jwst.__version__ >= "1.16", reason="requires jwst<1.16")
def test_slit_sign():

    import jwst.datamodels
    import jwst

    os.chdir(data_path())

    if jwst.__version__ < "100.16":
        file = "jw01345062001_03101_00001_nrs2_phot.138.1345_933.fits"
    else:
        file = "jw04233005001_03101_00002_nrs1_phot.140.4233_19489.fits"

    with jwst.datamodels.open(file) as slit:
        sign = utils.get_slit_sign(slit)
        assert sign == -1


def test_glob_sorted():

    import time

    for i in range(3):
        with open(f"dummy_file_{i}.txt", "w") as fp:
            time.sleep(1)
            fp.write(time.ctime() + "\n")
            if i == 1:
                fp.write(time.ctime() + "\n")

    files_by_date = utils.glob_sorted("dummy_file*txt", func=os.path.getmtime)

    files_by_date_rev = utils.glob_sorted(
        "dummy_file*txt", func=os.path.getmtime, reverse=True
    )

    files_by_size = utils.glob_sorted("dummy_file*txt", func=os.path.getsize)

    for file_set in [files_by_date, files_by_date_rev, files_by_size]:
        assert len(file_set) == 3

    assert files_by_date[0] == "dummy_file_0.txt"
    assert files_by_date_rev[0] == "dummy_file_2.txt"
    assert files_by_size[-1] == "dummy_file_1.txt"

    # Cleanup
    for f in files_by_size:
        os.remove(f)
