"""Forked functions from jwst.assign_wcs"""
import numpy as np

from jwst.assign_wcs import *

def validate_open_slits(input_model, open_slits, reference_files, use_sporder=None, get_bounding_boxes=False):
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
        if (xlow >= 2048 or ylow >= 2048 or
                xhigh <= 0 or yhigh <= 0 or
                xhigh - xlow < 2 or yhigh - ylow < 1):
            return False
        else:
            return True

    det2dms = dms_to_sca(input_model).inverse
    # read models from reference file
    disperser = DisperserModel(reference_files['disperser'])
    disperser = correct_tilt(disperser, input_model.meta.instrument.gwa_xtilt,
                             input_model.meta.instrument.gwa_ytilt)

    order, wrange = get_spectral_order_wrange(input_model,
                                              reference_files['wavelengthrange'])

    if use_sporder is not None:
        order = use_sporder

    input_model.meta.wcsinfo.waverange_start = wrange[0]
    input_model.meta.wcsinfo.waverange_end = wrange[1]
    input_model.meta.wcsinfo.spectral_order = order
    agreq = angle_from_disperser(disperser, input_model)
    # GWA to detector
    det2gwa = detector_to_gwa(reference_files,
                              input_model.meta.instrument.detector,
                              disperser)
    gwa2det = det2gwa.inverse
    # collimator to GWA
    collimator2gwa = collimator_to_gwa(reference_files, disperser)

    col2det = collimator2gwa & Identity(1) | Mapping((3, 0, 1, 2)) | agreq | \
        gwa2det | det2dms

    slit2msa = slit_to_msa(open_slits, reference_files['msa'])

    bounding_boxes = []

    for slit in slit2msa.slits:
        msa_transform = slit2msa.get_model(slit.name)
        msa2det = msa_transform & Identity(1) | col2det

        bb = compute_bounding_box(msa2det, wrange, slit.ymin, slit.ymax)

        valid = _is_valid_slit(bb)
        if not valid:
            log.info("Removing slit {0} from the list of open slits because the "
                     "WCS bounding_box is completely outside the detector.".format(slit.name))
            idx = np.nonzero([s.name == slit.name for s in open_slits])[0][0]
            open_slits.pop(idx)
        else:
            bounding_boxes.append(bb)

    if get_bounding_boxes:
        return open_slits, bounding_boxes
    else:
        return open_slits

def get_open_slits(input_model, reference_files=None, slit_y_range=[-.55, .55], validate=True, use_sporder=None):
    """Return the opened slits/shutters in a MOS or Fixed Slits exposure.
    """
    exp_type = input_model.meta.exposure.type.lower()
    lamp_mode = input_model.meta.instrument.lamp_mode
    if isinstance(lamp_mode, str):
        lamp_mode = lamp_mode.lower()
    else:
        lamp_mode = 'none'

    # MOS/MSA exposure requiring MSA metadata file
    if exp_type in ["nrs_msaspec", "nrs_autoflat"] or ((exp_type in ["nrs_lamp", "nrs_autowave"]) and
                                                       (lamp_mode == "msaspec")):
        prog_id = input_model.meta.observation.program_number.lstrip("0")
        msa_metadata_file, msa_metadata_id, dither_point = get_msa_metadata(input_model, reference_files)
        slits = get_open_msa_slits(prog_id, msa_metadata_file, msa_metadata_id, dither_point, slit_y_range)

    # Fixed slits exposure (non-TSO)
    elif exp_type == "nrs_fixedslit":
        slits = get_open_fixed_slits(input_model, slit_y_range)

    # Bright object (TSO) exposure in S1600A1 fixed slit
    elif exp_type == "nrs_brightobj":
        slits = [Slit('S1600A1', 3, 0, 0, 0, slit_y_range[0], slit_y_range[1], 5, 1)]

    # Lamp exposure using fixed slits
    elif exp_type in ["nrs_lamp", "nrs_autowave"]:
        if lamp_mode in ['fixedslit', 'brightobj']:
            slits = get_open_fixed_slits(input_model, slit_y_range)
    else:
        raise ValueError("EXP_TYPE {0} is not supported".format(exp_type.upper()))

    if reference_files is not None and slits:
        if validate:
            slits = validate_open_slits(input_model, slits, reference_files, use_sporder=use_sporder)
        log.info("Slits projected on detector {0}: {1}".format(input_model.meta.instrument.detector,
                                                               [sl.name for sl in slits]))
    if not slits:
        log_message = "No open slits fall on detector {0}.".format(input_model.meta.instrument.detector)
        log.critical(log_message)
        raise NoDataOnDetectorError(log_message)
    return slits


def zeroth_order_contamination_mask(input_model, reference_files=None, slit_y_range=[-0.55, 0.55], xsize=5, min_ysize=2, dq_flag="MSA_FAILED_OPEN", apply=True):
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

    if reference_files is None:
        step = AssignWcsStep()        
        reference_files = {}
        for k in step.reference_file_types:
            reference_files[k] = step.get_reference_file(input_model, k)

    dq = np.zeros((2048, 2048), dtype=np.uint32)

    try:
        open_slits = get_open_slits(
            input_model,
            reference_files,
            slit_y_range,
            validate=True,
            use_sporder=0
        )
    except NoDataOnDetectorError:
        return dq, [], []

    slits_, bounding_boxes = validate_open_slits(
        input_model,
        open_slits,
        reference_files,
        use_sporder=0,
        get_bounding_boxes=True
    )

    flag_ = jwst.datamodels.dqflags.pixel[dq_flag]
    
    for j, (slit_, bbox) in enumerate(zip(slits_, bounding_boxes)):
        
        xlim = [b+1.5 for b in bbox[0]]
        ylim = [b+1.5 for b in bbox[1]]
        
        x0 = np.mean(xlim)
        y0 = np.mean(ylim)
    
        ysize = np.maximum((ylim[1] - ylim[0])/2, min_ysize)

        slx = slice(
            np.maximum(int(np.round(x0 - xsize)), 0),
            int(np.round(x0 + xsize))
        )
        
        sly = slice(
            np.maximum(int(np.round(y0 - ysize)), 0),
            int(np.round(y0 + ysize))
        )           
        dq[sly, slx] |= pixel_flags[dq_flag]

    if apply:
        input_model.dq |= dq

    return dq, slits_, bounding_boxes

if 0:
    dq, slits_, bounding_boxes = zeroth_order_contamination_mask(
        input_model,
        reference_files=None,
        slit_y_range=[-0.55, 0.55],
        xsize=5,
        min_ysize=2,
        dq_flag="MSA_FAILED_OPEN"
    )