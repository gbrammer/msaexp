#!/usr/bin/env python

import sys
import os
import gc
import traceback
import glob
import yaml

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['grid.linestyle'] = ':'

import astropy.units as u
import astropy.io.fits as pyfits

import eazy
from grizli import utils
import grizli
import msaexp
import msaexp.spectrum

# test = """
# zfit="(0.1,9)"
# vel=150
#
# url=https://s3.amazonaws.com/msaexp-nirspec/extractions/rubies-egs63-v3/rubies-egs63-v3_prism-clear_4233_61168.spec.fits
#
# file=`echo $url | sed "s/-v3\// /" | awk '{print $2}'`; root=`echo $file | sed "s/-v3/ /" | awk '{print $1 "-v3"}'`; echo $file $root; python -c "import fit_msa_redshift; fit_msa_redshift.run_fit(**{'file': '$file', 'root': '$root', 's3_path': 'msaexp-nirspec/extractions', 'z0': $zfit, 'force_vel_width': $vel, 'scale_disp': 1.3})"; open `echo $file | sed "s/spec.fits/zfit.png/"`
#
# """

plt.ioff()

ORIG_TEMPLATES = None

__all__ = [
    "handle_nirspec_redshift",
    "run_one_redshift_fit",
]


def download_file(URL, force=False):
    """
    """
    import requests
    out_file = os.path.basename(URL)
    if os.path.exists(out_file) & (not force):
        return out_file

    msg = f'download {URL} to {out_file}'
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    resp = requests.get(URL.replace("+", "%2B"))
    # save to file        
    with open(out_file,'wb') as FLE:
        FLE.write(resp.content)

    return out_file

def upload_to_s3(file_name, bucket, object_name=None, ExtraArgs={}, verbose=True):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    import logging
    import boto3
    from botocore.exceptions import ClientError
    import os

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')

    msg = f'upload {file_name} to s3://{bucket}/{object_name}'
    
    if object_name.endswith('png'):
        ExtraArgs['ContentType'] = 'image/png'
    elif object_name.endswith('jpg'):
        ExtraArgs['ContentType'] = 'image/jpeg'
    elif object_name.endswith('fits'):
        ExtraArgs['ContentType'] = 'application/fits'
    else:
        ExtraArgs['ContentType'] = 'text/plain'
        
    try:
        response = s3_client.upload_file(
            file_name,
            bucket,
            object_name,
            ExtraArgs=ExtraArgs
        )

    except ClientError as e:
        logging.error(e)
        if verbose:
            print(msg + ' [FAILED]')
        return False

    if verbose:
        print(msg)

    return True


def read_templates(for_prism=True, eazy_templates=False, single_template=False):
    import eazy
    global ORIG_TEMPLATES
    
    # if not os.path.exists('templates'):
    #     eazy.symlink_eazy_inputs()
    
    # Preprocess templates
    ORIG_TEMPLATES = eazy.templates.read_templates_file(
        'templates/sfhz/agn_blue_sfhz_13.param'
    )
    
    #########
    # Add PSB template
    if not os.path.exists('young_post_starburst.fits'):
        UU = 'https://s3.amazonaws.com/msaexp-nirspec/scratch/young_post_starburst.fits'
        download_file(UU, force=False)

    psb = eazy.templates.Template('young_post_starburst.fits')
    ORIG_TEMPLATES.append(psb)
    
    if eazy_templates:
        return True
        
    files = """
#basis_templates/fsps_tau1.0_logz0.00_tage0.02_av0.00.fits
#basis_templates/fsps_tau1.0_logz0.00_tage0.02_av0.00_nolines.fits
#basis_templates/fsps_tau1.0_logz0.00_tage0.02_av0.00_noneb.fits
basis_templates/fsps_tau1.0_logz0.00_tage0.04_av0.00.fits
#basis_templates/fsps_tau1.0_logz0.00_tage0.16_av0.00.fits
basis_templates/fsps_tau1.0_logz0.00_tage0.69_av0.00.fits
basis_templates/fsps_tau1.0_logz0.00_tage3.09_av0.00.fits
#basis_templates/fsps_tau1.0_logz0.00_tage13.77_av0.00.fits
basis_templates/logz0.00_metal_lines.fits
basis_templates/logz0.00_nebular_continuum_H.fits
""".replace('logz0.00', 'logz0.00').split()
    
    ORIG_TEMPLATES = []
    for file in files:
        if '#' in file:
            continue
            
        ORIG_TEMPLATES.append(eazy.templates.Template(file))
    
    if 1:
        # Combine metal and recomb lines
        ORIG_TEMPLATES[-2].flux += ORIG_TEMPLATES[-1].flux
        _ = ORIG_TEMPLATES.pop(-1)
        ORIG_TEMPLATES[-1].name = ORIG_TEMPLATES[-1].name.replace('_metal', '_av0.00')
    
    # Dusty
    if for_prism:
        Avs = [0.5, 1.0, 2.0, 3.0]
    else:
        Avs = [0.5, 1.0, 2.0]
        
    if 1:
        import msaexp.resample_numba
        Alam = msaexp.resample_numba.calzetti2000_alambda(
            ORIG_TEMPLATES[0].wave/1.e4, 0
        )
        tred = []
        for Av in Avs:
            for t in ORIG_TEMPLATES:
                tr = eazy.templates.Template(
                    arrays=(t.wave, t.flux_flam()*10**(-0.4*Alam*Av)),
                    name=t.name.replace('av0.00', f'av{Av:.2f}')
                )
                tred.append(tr)
        
        ORIG_TEMPLATES += tred
    
    
    if for_prism:
        qso = eazy.templates.Template('temple_qsogen_z2.fits')        
        ORIG_TEMPLATES.append(qso)
        
        Alam = msaexp.resample_numba.smc_alambda(
            qso.wave/1.e4, 0
        )
        
        tred = []
        for Av in [0.5, 1.0]:
            tr = eazy.templates.Template(
                arrays=(qso.wave, qso.flux_flam()*10**(-0.4*Alam*Av)),
                name=qso.name + f'-av{Av:.2f}'
            )
            tred.append(tr)
        
        ORIG_TEMPLATES += tred
    
    if for_prism:
        qso = eazy.templates.Template('templates/sfhz/j0647agn+torus.fits')
        ORIG_TEMPLATES.append(qso)
        
    if single_template:
        ORIG_TEMPLATES = ORIG_TEMPLATES[2:3]
    
    # for t in ORIG_TEMPLATES:
    #     utils.log_comment(
    #         utils.LOGFILE,
    #         f'read_templates: {t.name} {len(t.wave)}',
    #         verbose=True
    #     )
        
    
EAZY_TEMPLATES = None

def resample_templates(R=800):
    global EAZY_TEMPLATES, ORIG_TEMPLATES
    
    if EAZY_TEMPLATES is not None:
        if EAZY_TEMPLATES[0].resample_R == R:
            return True
    
    if ORIG_TEMPLATES is None:
        read_templates()
        
    tlog = 10**np.arange(np.log10(900), np.log10(5.5e4), 1./R/np.log(10))
    
    msg = f'fit_msa_redshift: resample templates to R={R}  (N={len(tlog)})'
    utils.log_comment(utils.LOGFILE, msg, verbose=True)
    
    igm_obj = eazy.igm.Inoue14()
    z = 10
    igm = igm_obj.full_IGM(z, tlog*(1+z))

    EAZY_TEMPLATES = []
    for ti in ORIG_TEMPLATES:
        Rmin = (ti.wave / np.gradient(ti.wave)).min()
        
        if (len(tlog) < len(ti.wave)) | (Rmin < 800):
            EAZY_TEMPLATES.append(ti.resample(tlog, in_place=False))
            res_status = 'resampled'
        else:
            EAZY_TEMPLATES.append(ti)
            res_status = ''

        utils.log_comment(
            utils.LOGFILE,
            f'resample_templates: {ti.name:>44} {len(ti.wave):>8} {res_status}',
            verbose=True
        )
            
        # EAZY_TEMPLATES[-1].flux *= igm
        EAZY_TEMPLATES[-1].resample_R = R

# resample_templates(R=800)

def add_new_columns():
    """
    Add missing columns from nirspec_redshifts, e.g., new em lines
    """
    from grizli.aws import db
    zh = db.SQL("select ha.*, ne.grating from nirspec_redshift_handler ha, nirspec_extractions ne where ha.file like 'rubies-%%-v4%%prism-clear%%' and ha.file = ne.file order by ctime")
    zz = db.SQL("select ha.*, ne.grating from nirspec_redshifts ha, nirspec_extractions ne where ha.file like 'rubies-%%-v4%%prism-clear%%' and ha.file = ne.file")
    miss = ~np.isin(zh['file'], zz['file'])
    miss &= zh['zmin'] > 0.06
    zh = zh[miss]
    
    so = np.argsort(zh['zguess'])
    zh = zh[so]
    i = 0
    new_columns = []
    
    if 0:
        res = run_fit(
            file='rubies-uds21-nod-v3_prism-clear_4233_141828.spec.fits', 
            root='rubies-uds21-nod-v3',
            z0=[2, 4],
            clean=False
        )
    else:
        # i = 0
        kws = dict(zh[i]["file","root","zmin","zmax"])
        kws["clean"] = False
        kws["z0"] = [kws.pop(k) for k in ["zmin", "zmax"]]

        res = run_fit(**kws)

    file, ztab, zfit, row, rf_row  = res
    
    nrz = db.SQL("select * from nirspec_redshifts limit 1")
    
    for c in row:
        if (c not in nrz.colnames) & (c not in new_columns):
            print('column missing: ', c)
            new_columns.append(c)
            # SQL_COMMAND += f"ALTER TABLE nirspec_redshifts ADD COLUMN {c} real;\n"

    SQL_COMMAND = "ALTER TABLE nirspec_redshifts "
    SQL_COMMAND += ", ".join([f"ADD COLUMN {c} real" for c in new_columns])
    db.execute(SQL_COMMAND)
    
    for c in nrz.colnames:
        if c not in row:
            print('key missing: ', c)
            
    
def test():
    
    from msaexp.spectrum import make_templates
    file = 'rubies-egs61-v2_prism-clear_4233_75646.spec.fits'

    sampler = msaexp.spectrum.SpectrumSampler(file); spec = sampler.spec
    
    nspline = 23
    bspl = sampler.bspline_array(nspline=nspline, get_matrix=True)
    vel_width = 100
    scale_disp = 1.3
    use_full_dispersion = True
    kwargs = {}
    z = 4.9090
    
    templates, tline, _A = make_templates( sampler, z, bspl=bspl, eazy_templates=EAZY_TEMPLATES, vel_width=vel_width, scale_disp=scale_disp, use_full_dispersion=use_full_dispersion, disp=spec.disp, grating=spec.grating, **kwargs, )
    
    t = EAZY_TEMPLATES[0]
    
    tflam = sampler.resample_eazy_template( t, z=z, velocity_sigma=vel_width, scale_disp=scale_disp, fnu=False, )
    

def run_one_redshift_fit():
    """
    Query for an object to fit
    """
    import time
    from grizli.aws import db
    
    if False:
        zf = db.SQL("""select root, file, z from nirspec_redshifts where root LIKE 'rubies%%v2' and root not in ('rubies-egs2-v2', 'rubies-egs1-v2')
        order by file""")
        
        zf['status'] = 0
        zf['ctime'] = time.time()
        zf['zmin'] = np.maximum(zf['z'] - 0.1*(1 + zf['z']), 0.0)
        zf['zmax'] = zf['z'] + 0.1*(1 + zf['z'])
        zf['vel_width'] = -1.0
        zf['scale_disp'] = 1.3
        zf.remove_column('z')
        
        zf['file'] = [f.replace('v2','v3') for f in zf['file']]
        zf['root'] = [f.replace('v2','v3') for f in zf['root']]
        db.send_to_database('nirspec_redshift_handler', zf, if_exists='append')
        
        db.execute('CREATE INDEX on nirspec_redshift_handler (root, file)')
        
        # Bulk run on S3
        from grizli.aws.tile_mosaic import send_event_lambda, get_lambda_client
        from grizli.aws import db
        import time
        timeout = 30
        
        zh = db.SQL("""select * from nirspec_redshift_handler where status = 0 order by file""")
        nt1 = len(zh)

        NMAX = len(zh)

        istart = i = -1

        max_locked = 1600

        step = max_locked # - count_locked()[0]
        client = get_lambda_client()
        
        while i < NMAX-1:
            i+=1 
            # if tiles['tile'][i] == 1183:
            #     continue
    
            if i-istart == step:
                istart = i
                print(f'\n ############### \n {time.ctime()}: Pause for {timeout} s  / {step} run previously')
                time.sleep(timeout)
        
                # step = np.maximum(max_locked - count_locked()[0], 1)
                print(f'{time.ctime()}: Run {step} more \n ############## \n')
                
            event = dict(zh[i])
            event['status'] = int(event['status'])
            if 1:
                send_event_lambda(event, client=client, func='grizli-redshift-fit')
        
    obj = db.SQL("""
        SELECT * FROM nirspec_redshift_handler
        WHERE status = 0
        ORDER BY RANDOM()
        LIMIT 1
    """)
    
    if len(obj) == 0:
        with open('msa_redshift_finished.txt','w') as fp:
            fp.write(time.ctime() + '\n')
        
        return None
    
    result = handle_nirspec_redshift(dict(obj[0]))
    return result


def handle_nirspec_redshift(event, ACL='public-read', clean=True):
    """
    Run redshift fit for an object
    """
    import time
    from grizli.aws import db
    
    print(f'handle_nirspec_redshift: {event}')
    
    db.execute(f"""UPDATE nirspec_redshift_handler
    SET status = 1, ctime={time.time()}
    WHERE root = '{event['root']}' AND file = '{event['file']}'
    """
    )
    
    field = event['root'].split('-v')[0]
    if field in ['macs0417', 'macs0416', 'abell370', 'macs1423', 'macs1149']:
        s3_path = 'grizli-canucs/nirspec'
    else:
        s3_path = 'msaexp-nirspec/extractions'
        
    _ = run_fit(
        file=event['file'],
        root=event['root'],
        s3_path=s3_path,
        z0=[float(event['zmin']), float(event['zmax'])],
        force_vel_width=float(event['vel_width']),
        scale_disp=float(event['scale_disp'])
    )
    file, ztab, zfit, row, rf_row = _
    
    if row is None:
        # Failed
        db.execute(f"""UPDATE nirspec_redshift_handler
    SET status = 9, ctime={time.time()}
    WHERE root = '{event['root']}' AND file = '{event['file']}'
    """
        )
    else:
        db.execute(f"""
            UPDATE nirspec_redshift_handler
            SET status = 2, ctime={time.time()}
            WHERE root = '{event['root']}' AND file = '{event['file']}'
        """)
        
        db.execute(f"""
            DELETE FROM nirspec_redshifts
            WHERE root = '{event['root']}' AND file = '{event['file']}'
        """)
        
        trow = utils.GTable([row])
        db.send_to_database('nirspec_redshifts', trow, if_exists='append')
        
        if rf_row is not None:
            db.execute(f"""
                DELETE FROM nirspec_integrated
                WHERE root = '{event['root']}' AND file = '{event['file']}'
            """)
        
            trow = utils.GTable([rf_row])
            db.send_to_database('nirspec_integrated', trow, if_exists='append')
            
    # Upload results
    bucket = s3_path.split('/')[0]
    froot = file.split('.spec.fits')[0]
    files = glob.glob(froot + '*')
    files.sort()
    
    for fi in files:
        if fi == file:
            continue

        obj_name = os.path.join('/'.join(s3_path.split('/')[1:]), event['root'], fi)
        
        upload_to_s3(
            fi,
            bucket,
            object_name=obj_name,
            ExtraArgs={'ACL':ACL},
            verbose=True
        )
        
        if clean:
            os.remove(fi)

    if clean:
        os.remove(file)

    return row
    
def run_fit(file='rubies-egs61-v2_prism-clear_4233_75646.spec.fits', root='rubies-egs61-v2', s3_path='msaexp-nirspec/extractions', z0=[0.05, 16], force_vel_width=-1, scale_disp=1.3, clean=True, **kwargs):
    """
    """
    global EAZY_TEMPLATES
    
    outroot = root
    
    URL = f'https://s3.amazonaws.com/{s3_path}/{root}/{file}'
    utils.LOGFILE = file.replace('.spec.fits', '.zfit.log')

    utils.log_comment(utils.LOGFILE, f'grizli version {grizli.__version__}', verbose=True)
    utils.log_comment(utils.LOGFILE, f'msaexp version {msaexp.__version__}', verbose=True)
    
    download_file(URL, force=False)
    spec = msaexp.spectrum.SpectrumSampler(file)

    ##########
    # Emission line templates

    import yaml

    if 1:
        lwt = {'optred': [6564.697,
                      6549.86,
                      6585.27,
                          6718.29,
                      6732.67,
                      9071.1,
                      9533.2,
                      10052.2,
                      9548.65,
                      9231.6,
                      9017.44,
                      5877.249,
                      7067.1,
                      8446.7,
                      6679.995],
       'optblue': [4862.738,
                 4341.731,
                 4102.936,
                   3727.,
                 3890.191,
                 3971.236,
                 3968.59,
                 4960.295,
                 5008.24,
                 4364.436,
                 4687.5,
                 5877.249],
     'nir': [9533.2, 10941.2, 12821.7, 10832.057, 10833.306, 9548.65, 18756.3]}

        lrt = {'optred': [0.7658479337205669,
      0.22171784622413678,
      0.213351527571270517,
                          0.1,
      0.05598710736852652,
      0.010720880814820751,
      0.025599826996779446,
      0.021775468324444023,
      0.014644132155728325,
      0.01229073748547714,
      0.011048051620787565,
      0.046329604877447,
      0.04773331581780136,
      0.0030402114094973116,
      0.009847813846113357],
     'optblue': [0.11646892735360455,
      0.04721530033194111,
      0.03600225176448406,
                 0.1,
      0.017940195388501806,
      0.022559240041288113,
      0.01889883808879126,
      0.461/2.98,
      0.46164217872033314,
      0.02917002061981446,
      0.002156498957270335,
      0.022670079032331068],
     'nir': [0.021004558378288768,
      0.033969574802715,
      0.06865099322885622,
      0.1307938140847727,
      0.1307938140847727,
      0.014787245420594715,
            0.068*2.6]}
    
    lw, lr = utils.get_line_wavelengths()

    for k in ['Gal-UV-lines']:
        if k not in lwt:
            print(f'Add {k} to line lists')
            lwt[k] = [w for w in lw[k]]
            lrt[k] = [w for w in lr[k]]
        
    ############
    # Run it
    # scale_disp = 1.3
    nspline = 23
    use_eazy_templates = scale_disp > 0
    scale_disp = np.abs(scale_disp)
    
    plt_kwargs = dict(
        eazy_templates=EAZY_TEMPLATES,
        vel_width=(100 if force_vel_width < 0 else force_vel_width),
        scale_disp=scale_disp,
        nspline=nspline,
        Rline=2000,
        use_full_dispersion=True,
        is_prism=True,
        sys_err=0.02,
        ranges=((6200, 6800), (8800, 9600)),
        scale_uncertainty_kwargs={'order':1},
        plot_unit=u.microJansky,
    )

    zstep = None
    smooth_sigma = None

    ###########
    # Parse for gratings
    single_template = (z0[0] == 0.1) & (z0[1] == 0.8)
    
    if spec.meta['GRATING'].upper() == 'PRISM':
        msg = f'{file}: Prism templates'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        IS_GRATING = False
        
        read_templates(for_prism=True,
                       eazy_templates=use_eazy_templates,
                       single_template=single_template)
        
        resample_templates(R=800)
        plt_kwargs['eazy_templates'] = EAZY_TEMPLATES

    elif spec.meta['GRATING'].upper().endswith('M'):

        msg = f'{file}: Use Medium-resolution line widths'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
    
        plt_kwargs['vel_width'] = (100. if force_vel_width < 0 else force_vel_width)
        plt_kwargs['is_prism'] = False
        
        zstep = [float(np.maximum(plt_kwargs['vel_width']/3.e5*1.4, 0.001)),
                 float(plt_kwargs['vel_width']/3.e5/2.5)]

        # Emission line templates
        plt_kwargs['eazy_templates'] = [lwt, lrt]
        
        IS_GRATING = True
        if (file.startswith('bluejay')
            | file.startswith('valentino')
            | file.startswith('suspense-kriek')
            | file.startswith('cosmos-alpha')
            ) | (True):
            
            print(f'{file} use templates')
            
            read_templates(for_prism=False, eazy_templates=use_eazy_templates)
            
            resample_templates(R=2.5*3.e5/plt_kwargs['vel_width'])
            plt_kwargs['eazy_templates'] = EAZY_TEMPLATES
                
        if 'jades' not in outroot:
            smooth_sigma = 2

    elif spec.meta['GRATING'].upper().endswith('H'):

        msg = f'{file}: Use High-resolution line width'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        
        plt_kwargs['vel_width'] = (100. if force_vel_width < 0 else force_vel_width)
        plt_kwargs['is_prism'] = False

        zstep = [float(np.maximum(plt_kwargs['vel_width']/3.e5*1.4, 0.001)),
                 float(plt_kwargs['vel_width']/3.e5/5.)]

        plt_kwargs['eazy_templates'] = EAZY_TEMPLATES
    
        # Emission line templates
        plt_kwargs['eazy_templates'] = [lwt, lrt]

        if 'cecilia' in outroot:
            z0 = [1.1, 5]
        
        smooth_sigma = 3
        IS_GRATING = True
        # resample_templates(R=4000)
    
    orig_nspline = nspline
    if (plt_kwargs['eazy_templates'] is not None):
        if len(plt_kwargs['eazy_templates']) == 1:
            orig_nspline = plt_kwargs['nspline']
            plt_kwargs['nspline'] = 7
            
    EXTRA_KWS = {}

    use_aper_columns = True

    plt_kwargs['use_aper_columns'] = use_aper_columns

    plt_kwargs['halpha_prism'] = ['Ha+NII','CIV-1549', 'HeII-1640', 'NIV-1487']

    msaexp.spectrum.SCALE_UNCERTAINTY = np.zeros(2)

    plt.ioff()

    try:
        _ = msaexp.spectrum.fit_redshift(file=file, z0=z0, zstep=zstep, **plt_kwargs)
        fig0, ztab, zfit = _
    except:
        utils.log_exception(utils.LOGFILE, traceback)
        return file, None, None, None, None
        
    ########
    # remake figures
    ymax = None

    with pyfits.open(file) as outhdu:
        fig = msaexp.utils.drizzled_hdu_figure(
            outhdu,
            unit='fnu',
            smooth_sigma=smooth_sigma,
            z=zfit['z'],
            use_aper_columns=use_aper_columns,
            ymax=ymax,
            ymax_sigma_scale=10+10*IS_GRATING,
        )
        
        ax = fig.axes[2]
        utils.figure_timestamp(fig)
        fig.savefig(file.replace('.spec.fits','.fnu.png'))

        fig = msaexp.utils.drizzled_hdu_figure(
            outhdu,
            unit='flam',
            smooth_sigma=smooth_sigma,
            z=zfit['z'],
            use_aper_columns=use_aper_columns,
            ymax=ymax,
            ymax_sigma_scale=10+10*IS_GRATING,
        )
        utils.figure_timestamp(fig)    
        fig.savefig(file.replace('.spec.fits','.flam.png'))
    
    # Row for table
    plt_kwargs['eazy_templates'] = None
    plt_kwargs['nspline'] = orig_nspline
    print(plt_kwargs['nspline'])
    _fig, zsp, res = msaexp.spectrum.plot_spectrum(file, z=zfit['z'], **plt_kwargs)
    
    xres = {}
    for k in res:
        if k in ['wave','flux','err','escale','model','mline','templates','covar']:
            continue
        elif k in ['eqwidth']:
            for ki in res[k]:
                knew = ki.replace(' ','_').replace('-','_').lower().replace('line','eqw')
                xres[knew] = res[k][ki]
            
        elif k in ['coeffs']:
            for ki in res[k]:
                knew = ki.replace(' ','_').replace('-','_').lower()
                xres[knew], xres[knew+'_err'] = res[k][ki]
        else:
            xres[k] = res[k]
    
    xres['escale0'], xres['escale1'] = msaexp.spectrum.SCALE_UNCERTAINTY
    row = {}
    for k in xres:
        if k in ['ra','dec','label','name']:
            continue

        if isinstance(xres[k], str):
            row[k] = xres[k]
            continue
            
        if hasattr(xres[k], '__len__'):
            val = xres[k][0]
        else:
            val = xres[k]
            
        row[k] = float(val) if np.isreal(val) else val
        
    sn_line = 0
    line_at_max = '------------'

    for c in row:
        if c.startswith('line_') & (not c.endswith('_err')):
            if row[c + '_err'] == 0:
                continue
            
            sn_i = row[c] / row[c + '_err']
            if sn_i > sn_line:
                sn_line = sn_i
                line_at_max = c.split('line_')[-1]
    
    row['sn_line'] = sn_line
    # row['max_line'] = line_at_max
    row['root'] = root
    row['ctime'] = os.path.getmtime(file)
    # row['status'] = 2
    row['dof'] = int(row['dof'])
    
    for c in list(row.keys()):
        if '.' in c:
            print(c)
            row[c.replace('.','p')] = row.pop(c)
        if '+' in c:
            print(c)
            row[c.replace('+','_')] = row.pop(c)
    
    with open(file.replace('.spec.fits', '.zfit.yaml.row'),'w') as fp:
        yaml.dump(row, fp)
    
    del(fig); del(_fig); del(fig0); del(res)
    
    ######
    # Integrated filters
    
    _SCALE_KWARGS = dict(
        order=plt_kwargs['scale_uncertainty_kwargs']['order'],
        sys_err=plt_kwargs['sys_err'],
        nspline=nspline,
        scale_disp=scale_disp,
        vel_width=plt_kwargs['vel_width'],
        initial_mask=(0.3, 5),
        fit_sys_err=False,
    )

    rf_row, rf_sed = msaexp.spectrum.do_integrate_filters(
        file,
        z=zfit['z'],
        scale_kwargs=_SCALE_KWARGS
    )

    rf_row['root'] = root
    if 'z' in rf_row:
        rf_row['zrf'] = rf_row.pop('z')

    del(rf_sed)
    
    gc.collect()
    plt.close('all')
    
    return file, ztab, zfit, row, rf_row

if __name__ == '__main__':
    
    import eazy
    print(os.getcwd())
    
    # if not os.path.exists('templates'):
    #     eazy.symlink_eazy_inputs()
    #
    # print('templates: ', os.path.exists('templates'))
    
    run_one_redshift_fit()
    