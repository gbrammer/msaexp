"""Forked functions from jwst.assign_wcs"""

import logging
import numpy as np
import copy

from astropy.modeling import models
from astropy.modeling.models import (
    Mapping,
    Identity,
    Const1D,
    Scale,
    Tabular1D,
)
from astropy import units as u
from astropy import coordinates as coord
from astropy.io import fits
from gwcs import coordinate_frames as cf
from gwcs.wcstools import grid_from_bounding_box

from stdatamodels.jwst.datamodels import (
    CollimatorModel,
    CameraModel,
    DisperserModel,
    FOREModel,
    IFUFOREModel,
    MSAModel,
    OTEModel,
    IFUPostModel,
    IFUSlicerModel,
    WavelengthrangeModel,
    FPAModel,
)
from stdatamodels.jwst.transforms.models import (
    Rotation3DToGWA,
    DirCos2Unitless,
    Slit2Msa,
    AngleFromGratingEquation,
    WavelengthFromGratingEquation,
    Gwa2Slit,
    Unitless2DirCos,
    Logical,
    Slit,
    Snell,
    RefractionIndexFromPrism,
)

from jwst.assign_wcs.util import (
    MSAFileError,
    NoDataOnDetectorError,
    not_implemented_mode,
    velocity_correction,
)
from jwst.assign_wcs import pointing
from jwst.lib.exposure_types import is_nrs_ifu_lamp

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# from jwst.assign_wcs.nirspec import *
import jwst.assign_wcs.nirspec
from jwst.assign_wcs.nirspec import (
    get_msa_metadata,
    get_open_msa_slits,
    dms_to_sca,
    correct_tilt,
    angle_from_disperser,
    detector_to_gwa,
    collimator_to_gwa,
    slit_to_msa,
    compute_bounding_box,
    get_open_fixed_slits,
    create_frames,
    get_spectral_order_wrange,
    get_disperser,
    gwa_to_ifuslit,
    ifuslit_to_slicer,
    slicer_to_msa,
    oteip_to_v23,
    ifu_msa_to_oteip
)


def validate_open_slits(
    input_model,
    open_slits,
    reference_files,
    use_sporder=None,
    get_bounding_boxes=False,
):
    """
    Remove slits which do not project on the detector from the list of open slits.
    For each slit computes the transform from the slit to the detector and
    determines the bounding box.

    Parameters
    ----------
    input_model : jwst.datamodels.JwstDataModel
        Input data model

    use_sporder : None, int
        Force spectral order to use

    Returns
    -------
    slit2det : dict
        A dictionary with the slit to detector transform for each slit,
        {slit_id: astropy.modeling.Model}
    """

    def _is_valid_slit(domain):
        xlow, xhigh = domain[0]
        ylow, yhigh = domain[1]
        if (
            xlow >= 2048
            or ylow >= 2048
            or xhigh <= 0
            or yhigh <= 0
            or xhigh - xlow < 2
            or yhigh - ylow < 1
        ):
            return False
        else:
            return True

    det2dms = dms_to_sca(input_model).inverse
    # read models from reference file
    disperser = DisperserModel(reference_files["disperser"])
    disperser = correct_tilt(
        disperser,
        input_model.meta.instrument.gwa_xtilt,
        input_model.meta.instrument.gwa_ytilt,
    )

    order, wrange = get_spectral_order_wrange(
        input_model, reference_files["wavelengthrange"]
    )

    if use_sporder is not None:
        order = use_sporder

    input_model.meta.wcsinfo.waverange_start = wrange[0]
    input_model.meta.wcsinfo.waverange_end = wrange[1]
    input_model.meta.wcsinfo.spectral_order = order
    agreq = angle_from_disperser(disperser, input_model)
    # GWA to detector
    det2gwa = detector_to_gwa(
        reference_files, input_model.meta.instrument.detector, disperser
    )
    gwa2det = det2gwa.inverse
    # collimator to GWA
    collimator2gwa = collimator_to_gwa(reference_files, disperser)

    col2det = (
        collimator2gwa & Identity(1)
        | Mapping((3, 0, 1, 2))
        | agreq
        | gwa2det
        | det2dms
    )

    slit2msa = slit_to_msa(open_slits, reference_files["msa"])

    bounding_boxes = []

    for slit in slit2msa.slits:
        msa_transform = slit2msa.get_model(slit.name)
        msa2det = msa_transform & Identity(1) | col2det

        bb = compute_bounding_box(msa2det, wrange, slit.ymin, slit.ymax)

        valid = _is_valid_slit(bb)
        if not valid:
            log.info(
                "Removing slit {0} from the list of open slits because the "
                "WCS bounding_box is completely outside the detector.".format(
                    slit.name
                )
            )
            idx = np.nonzero([s.name == slit.name for s in open_slits])[0][0]
            open_slits.pop(idx)
        else:
            bounding_boxes.append(bb)

    if get_bounding_boxes:
        return open_slits, bounding_boxes
    else:
        return open_slits


def get_open_slits(
    input_model,
    reference_files=None,
    slit_y_range=[-0.55, 0.55],
    validate=True,
    use_sporder=None,
):
    """Return the opened slits/shutters in a MOS or Fixed Slits exposure."""
    exp_type = input_model.meta.exposure.type.lower()
    lamp_mode = input_model.meta.instrument.lamp_mode
    if isinstance(lamp_mode, str):
        lamp_mode = lamp_mode.lower()
    else:
        lamp_mode = "none"

    # MOS/MSA exposure requiring MSA metadata file
    if exp_type in ["nrs_msaspec", "nrs_autoflat"] or (
        (exp_type in ["nrs_lamp", "nrs_autowave"]) and (lamp_mode == "msaspec")
    ):
        prog_id = input_model.meta.observation.program_number.lstrip("0")
        msa_metadata_file, msa_metadata_id, dither_point = get_msa_metadata(
            input_model, reference_files
        )
        slits = get_open_msa_slits(
            prog_id,
            msa_metadata_file,
            msa_metadata_id,
            dither_point,
            slit_y_range,
        )

    # Fixed slits exposure (non-TSO)
    elif exp_type == "nrs_fixedslit":
        slits = get_open_fixed_slits(input_model, slit_y_range)

    # Bright object (TSO) exposure in S1600A1 fixed slit
    elif exp_type == "nrs_brightobj":
        slits = [
            Slit("S1600A1", 3, 0, 0, 0, slit_y_range[0], slit_y_range[1], 5, 1)
        ]

    # Lamp exposure using fixed slits
    elif exp_type in ["nrs_lamp", "nrs_autowave"]:
        if lamp_mode in ["fixedslit", "brightobj"]:
            slits = get_open_fixed_slits(input_model, slit_y_range)
    else:
        raise ValueError(
            "EXP_TYPE {0} is not supported".format(exp_type.upper())
        )

    if reference_files is not None and slits:
        if validate:
            slits = validate_open_slits(
                input_model, slits, reference_files, use_sporder=use_sporder
            )
        log.info(
            "Slits projected on detector {0}: {1}".format(
                input_model.meta.instrument.detector, [sl.name for sl in slits]
            )
        )
    if not slits:
        log_message = "No open slits fall on detector {0}.".format(
            input_model.meta.instrument.detector
        )
        log.critical(log_message)
        raise NoDataOnDetectorError(log_message)
    return slits


def zeroth_order_mask(
    input_model,
    reference_files=None,
    slit_y_range=[-0.55, 0.55],
    xsize=5,
    min_ysize=3,
    dq_flag="MSA_FAILED_OPEN",
    apply=True,
    **kwargs,
):
    """
    Add mask for zeroth order contamination to dq array.

    Parameters
    ----------
    input_model : `jwsts.datamodels.ImageModel`
        Data model for a NIRSpec MSA exposure

    reference_files : dict, None
        Reference filenames.  If None, then will generate from `jwst.assign_wcs.AssignWcsStep.reference_file_types`.

    slit_y_range : [float, float]
        Slit coordinates

    xsize : scalar
        Size of mask in dispersion axis relative to the center of the computed bounding box

    min_ysize : scalar
        Minimum size of the mask in the cross-dispersion axis

    dq_flag : str
        Flag name in `jwst.datamodels.dqflags.pixel` to use for DQ bit

    apply : bool
        Add updated DQ mask to ``input_model.dq``

    Returns
    -------
    dq : 2D array
        DQ array for contaminated pixels

    slits : list
        List of `stdatamodels.jwst.transforms.models.Slit` models for open slits used for the mask

    bounding_boxes : list
        List of ``((xmin, xmax), (ymin, ymax))`` bounding boxes computed by `jwst.assign_wcs.nirspec.validate_open_slits`

    """
    from jwst.assign_wcs.nirspec import slitlets_wcs, get_spectral_order_wrange
    from jwst.assign_wcs import AssignWcsStep
    from jwst.assign_wcs.util import NoDataOnDetectorError

    from jwst.datamodels.dqflags import pixel as pixel_flags
    import jwst.datamodels

    dq = np.zeros((2048, 2048), dtype=np.uint32)

    # Only needed for NRS1 and M gratings
    inst_mode = "{grating}-{detector}".format(
        **input_model.meta.instrument.instance
    )
    valid_inst_modes = [
        "G140M-NRS1",
        "G235M-NRS1",
        "G395M-NRS1",
    ]
    if inst_mode not in valid_inst_modes:
        log.info(
            f"{__name__}: mode = {inst_mode}, zeroth order mask only needed for {' '.join(valid_inst_modes)}"
        )
        return dq, [], []

    log.info(
        f"{__name__}: slit_y_range={slit_y_range}, xsize={xsize}, min_ysize={min_ysize} apply={apply}"
    )

    if reference_files is None:
        step = AssignWcsStep()
        reference_files = {}
        for k in step.reference_file_types:
            reference_files[k] = step.get_reference_file(input_model, k)

    try:
        open_slits = get_open_slits(
            input_model,
            reference_files,
            slit_y_range,
            validate=True,
            use_sporder=0,
        )
    except NoDataOnDetectorError:
        return dq, [], []

    slits_, bounding_boxes = validate_open_slits(
        input_model,
        open_slits,
        reference_files,
        use_sporder=0,
        get_bounding_boxes=True,
    )

    log.info(f"{__name__}: Set DQ={dq_flag} for {len(slits_)} slits")

    flag_ = jwst.datamodels.dqflags.pixel[dq_flag]

    for j, (slit_, bbox) in enumerate(zip(slits_, bounding_boxes)):

        xlim = [b + 1.5 for b in bbox[0]]
        ylim = [b + 1.5 for b in bbox[1]]

        x0 = np.mean(xlim)
        y0 = np.mean(ylim)

        ysize = np.maximum((ylim[1] - ylim[0]) / 2, min_ysize)

        slx = slice(
            np.maximum(int(np.round(x0 - xsize)), 0), int(np.round(x0 + xsize))
        )

        sly = slice(
            np.maximum(int(np.round(y0 - ysize)), 0), int(np.round(y0 + ysize))
        )
        dq[sly, slx] |= pixel_flags[dq_flag]

    if apply:
        input_model.dq |= dq

    return dq, slits_, bounding_boxes


# if 0:
#     dq, slits_, bounding_boxes = zeroth_order_mask(
#         input_model,
#         reference_files=None,
#         slit_y_range=[-0.55, 0.55],
#         xsize=5,
#         min_ysize=2,
#         dq_flag="MSA_FAILED_OPEN",
#     )

#
# Overwrite ifu processing to include all detectors
#

def ifu(input_model, reference_files, slit_y_range=[-.55, .55], limit_detectors=False):
    """
    The Nirspec IFU WCS pipeline.

    The coordinate frames are:
    "detector" : the science frame
    "sca" : frame associated with the SCA
    "gwa" " just before the GWA going from detector to sky
    "slit_frame" : frame associated with the virtual slit
    "slicer' : frame associated with the slicer
    "msa_frame" : at the MSA
    "oteip" : after the FWA
    "v2v3" and "world"

    Parameters
    ----------
    input_model : `~jwst.datamodels.JwstDataModel`
        The input data model.
    reference_files : dict
        The reference files used for this mode.
    slit_y_range : list
        The slit dimensions relative to the center of the slit.
    """
    detector = input_model.meta.instrument.detector
    grating = input_model.meta.instrument.grating
    filter = input_model.meta.instrument.filter
    # Check for ifu reference files
    if reference_files['ifufore'] is None and \
       reference_files['ifuslicer'] is None and \
       reference_files['ifupost'] is None:
        # No ifu reference files, won't be able to create pipeline
        log_message = 'No ifufore, ifuslicer or ifupost reference files'
        log.critical(log_message)
        raise RuntimeError(log_message)
    # Check for data actually being present on NRS2
    log_message = "No IFU slices fall on detector {0}".format(detector)

    if limit_detectors:
        if detector == "NRS2" and grating.endswith('M'):
            # Mid-resolution gratings do not project on NRS2.
            log.critical(log_message)
            raise NoDataOnDetectorError(log_message)
        if detector == "NRS2" and grating == "G140H" and filter == "F070LP":
            # This combination of grating and filter does not project on NRS2.
            log.critical(log_message)
            raise NoDataOnDetectorError(log_message)

    slits = np.arange(30)
    # Get the corrected disperser model
    disperser = get_disperser(input_model, reference_files['disperser'])

    # Get the default spectral order and wavelength range and record them in the model.
    sporder, wrange = get_spectral_order_wrange(input_model, reference_files['wavelengthrange'])
    input_model.meta.wcsinfo.waverange_start = wrange[0]
    input_model.meta.wcsinfo.waverange_end = wrange[1]
    input_model.meta.wcsinfo.spectral_order = sporder

    # DMS to SCA transform
    dms2detector = dms_to_sca(input_model)
    # DETECTOR to GWA transform
    det2gwa = Identity(2) & detector_to_gwa(reference_files,
                                            input_model.meta.instrument.detector,
                                            disperser)

    # GWA to SLIT
    gwa2slit = gwa_to_ifuslit(slits, input_model, disperser, reference_files, slit_y_range)

    # SLIT to MSA transform
    slit2slicer = ifuslit_to_slicer(slits, reference_files, input_model)

    # SLICER to MSA Entrance
    slicer2msa = slicer_to_msa(reference_files)

    det, sca, gwa, slit_frame, msa_frame, oteip, v2v3, v2v3vacorr, world = create_frames()

    exp_type = input_model.meta.exposure.type.upper()

    is_lamp_exposure = exp_type in ['NRS_LAMP', 'NRS_AUTOWAVE', 'NRS_AUTOFLAT']

    if input_model.meta.instrument.filter == 'OPAQUE' or is_lamp_exposure:
        # If filter is "OPAQUE" or if internal lamp exposure the NIRSPEC WCS pipeline stops at the MSA.
        pipeline = [(det, dms2detector),
                    (sca, det2gwa.rename('detector2gwa')),
                    (gwa, gwa2slit.rename('gwa2slit')),
                    (slit_frame, slit2slicer),
                    ('slicer', slicer2msa),
                    (msa_frame, None)]
    else:
        # MSA to OTEIP transform
        msa2oteip = ifu_msa_to_oteip(reference_files)
        # OTEIP to V2,V3 transform
        # This includes a wavelength unit conversion from meters to microns.
        oteip2v23 = oteip_to_v23(reference_files, input_model)

        # Compute differential velocity aberration (DVA) correction:
        va_corr = pointing.dva_corr_model(
            va_scale=input_model.meta.velocity_aberration.scale_factor,
            v2_ref=input_model.meta.wcsinfo.v2_ref,
            v3_ref=input_model.meta.wcsinfo.v3_ref
        ) & Identity(1)

        # V2, V3 to sky
        tel2sky = pointing.v23tosky(input_model) & Identity(1)

        # Create coordinate frames in the NIRSPEC WCS pipeline"
        #
        # The oteip2v2v3 transform converts the wavelength from meters (which is assumed
        # in the whole pipeline) to microns (which is the expected output)
        #
        # "detector", "gwa", "slit_frame", "msa_frame", "oteip", "v2v3", "world"

        pipeline = [(det, dms2detector),
                    (sca, det2gwa.rename('detector2gwa')),
                    (gwa, gwa2slit.rename('gwa2slit')),
                    (slit_frame, slit2slicer),
                    ('slicer', slicer2msa),
                    (msa_frame, msa2oteip.rename('msa2oteip')),
                    (oteip, oteip2v23.rename('oteip2v23')),
                    (v2v3, va_corr),
                    (v2v3vacorr, tel2sky),
                    (world, None)]

    return pipeline
    
jwst.assign_wcs.nirspec.ifu = ifu