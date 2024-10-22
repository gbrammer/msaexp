import os
import copy

import numpy as np
import matplotlib.pyplot as plt

import pytest
import jwst

from .. import utils, pipeline
from .. import drizzle as msadrizzle

pipe = None

# TARGETS = ["1345_933"]

if jwst.__version__ < "100.16":
    TARGETS = ["1345_933"]
else:
    TARGETS = ["4233_19489"]

if jwst.__version__ >= "1.16":
    pytest.skip(
        "Skip drizzle with jwst={jwst.__version__} >= 1.16",
        allow_module_level=True,
    )


def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


def test_combine():
    """
    Test drizzle combination from slitlet
    """
    os.chdir(data_path())

    drizzle_kws = dict(
        center_on_source=False,  # Center output source as defined in MSA file.
        fix_slope=None,  # Cross-dispersion scale dslit/dpix
        force_nypix=31,  # Y size of output array
        bkg_offset=6,  # Number of pixels to roll for background sub
        bkg_parity=[-1, 1],  # Roll directions for background, i.e., -1 rolls
        log_step=False,  # Log wavelength steps
        outlier_threshold=10,  # Outlier rejection threshold
        err_threshold=1000,  # Reject pixels in a slit where err > err_threshold*median(err)
        show_drizzled=True,  # Figures
        show_slits=True,
        imshow_kws=dict(
            vmin=-0.05, vmax=0.4, aspect="auto", cmap="cubehelix_r"
        ),
        sn_threshold=-np.inf,  # Reject pixels where S/N < sn_threshold.
        bar_threshold=0.0,  # Mask pixels where barshadow array less than this value
    )

    DRIZZLE_PARAMS = copy.deepcopy(msadrizzle.DRIZZLE_PARAMS)
    DRIZZLE_PARAMS["kernel"] = "square"
    DRIZZLE_PARAMS["pixfrac"] = 1.0

    outroot = "test-driz"
    target = "933"

    _ = msadrizzle.drizzle_slitlets(
        target,
        output=outroot,
        # files=slit_files[:],
        **drizzle_kws,
    )

    figs, hdu_data, wavedata, all_slits, drz_data = _

    # Fixed "slope"
    drizzle_kws["fix_slope"] = 0.2
    outroot = "test-driz-slope"
    _ = msadrizzle.drizzle_slitlets(
        target,
        output=outroot,
        # files=slit_files[:],
        **drizzle_kws,
    )

    figs, hdu_data, wavedata, all_slits, drz_data = _

    # Center on source
    drizzle_kws["center_on_source"] = True
    outroot = "test-driz-center"
    _ = msadrizzle.drizzle_slitlets(
        target,
        output=outroot,
        # files=slit_files[:],
        **drizzle_kws,
    )

    figs, hdu_data, wavedata, all_slits, drz_data = _

    # 1D extraction
    extract_kws = dict(
        prf_sigma=1.2,
        fix_sigma=False,
        prf_center=None,
        fix_center=False,
        center_limit=3,
        verbose=True,
        profile_slice=None,
    )

    grating = "prism-clear"
    hdul = hdu_data[grating]

    outhdu = msadrizzle.extract_from_hdul(hdul, **extract_kws)

    file = f"{outroot}_{target}.spec.fits"

    outhdu.writeto(file, overwrite=True)

    # Make figures
    fig = utils.drizzled_hdu_figure(outhdu, unit="fnu")
    fig.savefig(f"{outroot}-{grating}_{target}.fnu.png")

    fig = utils.drizzled_hdu_figure(outhdu, unit="flam")
    fig.savefig(f"{outroot}-{grating}_{target}.flam.png")

    plt.close("all")


#############
# A-B background subtractions


def test_init_pipeline():
    """
    Initialize pipeline object with previously-extracted slitlets
    """
    import glob

    global pipe

    os.chdir(data_path())

    mode = "jw01345062001_03101_00001_nrs2"

    files = [f"jw01345062001_03101_0000{i}_nrs2_rate.fits" for i in [1, 2, 3]]

    pipe = pipeline.NirspecPipeline(mode=mode, files=files)

    pipe.full_pipeline(
        run_extractions=False,
        initialize_bkg=True,
        targets=TARGETS,
        load_saved="phot",
    )


def test_extract_spectra_pipeline():
    """
    Extract single spectra
    """

    os.chdir(data_path())

    fit_profile = {"min_delta": 20}
    yoffset = 0.01

    key = TARGETS[0]

    _data = pipe.extract_spectrum(
        key,
        skip=[],
        yoffset=yoffset,
        prof_sigma=0.7,
        trace_sign=-1,
        fit_profile_params=fit_profile,
    )

    plt.close("all")

    slitlet, sep1d, opt1d, fig = _data


def test_drizzle_combine_slits():
    """
    Drizzle combination
    """

    os.chdir(data_path())

    DRIZZLE_PARAMS = dict(
        output=None,
        single=True,
        blendheaders=True,
        pixfrac=0.6,
        kernel="square",
        fillval=0,
        wht_type="ivm",
        good_bits=0,
        pscale_ratio=1.0,
        pscale=None,
        verbose=False,
    )

    key = TARGETS[0]

    slits = []
    slits += pipe.get_background_slits(key, step="bkg", check_background=True)

    for slit in slits:
        slit.dq = slit.dq & (1 + 1024)

    print(f"{key}  N= {len(slits)}  slits")

    drizzle_kws = dict(
        center_on_source=False,  # Center output source as defined in MSA file.
        fix_slope=None,  # Cross-dispersion scale dslit/dpix
        force_nypix=31,  # Y size of output array
        bkg_offset=-6,  # Number of pixels to roll for background sub
        bkg_parity=[-1, 1],  # Roll directions for background, i.e., -1 rolls
        log_step=False,  # Log wavelength steps
        outlier_threshold=10,  # Outlier rejection threshold
        err_threshold=1000,  # Reject pixels in a slit where err > err_threshold*median(err)
        show_drizzled=True,  # Figures
        show_slits=True,
        imshow_kws=dict(
            vmin=-0.05, vmax=0.4, aspect="auto", cmap="cubehelix_r"
        ),
        sn_threshold=-np.inf,  # Reject pixels where S/N < sn_threshold.
        bar_threshold=0.0,  # Mask pixels where barshadow array less than this value
    )

    DRIZZLE_PARAMS = copy.deepcopy(msadrizzle.DRIZZLE_PARAMS)
    DRIZZLE_PARAMS["kernel"] = "square"
    DRIZZLE_PARAMS["pixfrac"] = 1.0

    target = "933"

    drizzle_kws["fix_slope"] = 0.2
    drizzle_kws["center_on_source"] = True
    outroot = "test-driz-center-bkg"

    _ = msadrizzle.drizzle_slitlets(
        target,
        files=slits,
        output=outroot,
        # files=slit_files[:],
        **drizzle_kws,
    )

    figs, hdu_data, wavedata, all_slits, drz_data = _

    # 1D extraction
    extract_kws = dict(
        prf_sigma=1.2,
        fix_sigma=False,
        prf_center=None,
        fix_center=False,
        center_limit=3,
        verbose=True,
        profile_slice=None,
    )

    grating = "prism-clear"
    hdul = hdu_data[grating]

    outhdu = msadrizzle.extract_from_hdul(hdul, **extract_kws)

    file = f"{outroot}_{target}.spec.fits"

    outhdu.writeto(file, overwrite=True)

    # Make figures
    fig = utils.drizzled_hdu_figure(outhdu, unit="fnu")
    fig.savefig(f"{outroot}-{grating}_{target}.fnu.png")

    fig = utils.drizzled_hdu_figure(outhdu, unit="flam")
    fig.savefig(f"{outroot}-{grating}_{target}.flam.png")

    plt.close("all")

    # #########
    # hdul = utils.drizzle_2d_pipeline(slits,
    #                                  drizzle_params=DRIZZLE_PARAMS,
    #                                  fit_prf=True,
    #                                  outlier_threshold=30000,
    #                                  prf_center=-0.0,
    #                                  prf_sigma=1.0,
    #                                  fix_sigma=True,
    #                                  center_limit=6.0,
    #                                  standard_waves=False,
    #                                  # profile_slice=slice(100,150),
    #                                  )
    #
    # z = 4.2341
    # _fig = utils.drizzled_hdu_figure(hdul,
    #                                  z=z,
    #                                  xlim=None,
    #                                  unit='fnu')
    # ax = _fig.axes[2]
    # xl = ax.get_xlim()
    #
    # ax.text(0.02, 0.82, key, ha='left', va='bottom', transform=ax.transAxes)
    #
    # plt.close('all')
    # froot = 'ceers-prism'
    # hdul.writeto(f'{froot}.{key}.v0.spec.fits', overwrite=True)
    #
    # # Figure
    # with pyfits.open(f'{froot}.{key}.v0.spec.fits') as outhdu:
    #     fig = utils.drizzled_hdu_figure(outhdu, unit='fnu')
    #     fig.savefig(f'{froot}.{key}.v0.spec.fnu.png')
