#!/usr/bin/env python
"""
NIRSpec preprocessing steps up to slitlet extractions
"""

import sys

# Quiet!
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CRDS_CONTEXT'] = os.environ['CRDS_CTX'] = 'jwst_1225.pmap'
# REDUCTION_VERSION = '-v3'

from grizli import jwst_level1
from msaexp import pipeline
import mastquery.jwst
import mastquery.utils

import os
import glob
import yaml
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

import astropy.time

import grizli
from grizli import utils, jwst_utils
jwst_utils.set_quiet_logging()
utils.set_warnings()

import astropy.io.fits as pyfits
from jwst.datamodels import SlitModel

import msaexp
from msaexp import pipeline
try:
    from msaexp import pipeline_extended
    HAS_EXTENDED_PIPELINE = True
except:
    HAS_EXTENDED_PIPELINE = False

import jwst
import time

from msaexp import slit_group
from msaexp import utils as msautils

NIRSPEC_HOME = '/home/ec2-user/telescopes/NIRSpec'
HOME = '/GrizliImaging/'

if not os.path.exists(HOME):
    print(f"HOME={HOME} not found, using HOME={os.getcwd()}")
    HOME = os.getcwd()

# NIRSPEC_HOME = '/GrizliImaging/NIRSpec'

def test():

    go = """
    preprocess_nirspec_file(
        rate_file='jw01286005001_09101_00001_nrs1_rate.fits'
        root='jades-gds05-v4'
        as_fixed=False
        rename_f070=True
        context='jwst_1293.pmap'
    )
    """

def sync():
    bash = """

dirs=`ls | grep v4`
for dir in $dirs; do
    echo s3://msaexp-nirspec/extractions/slitlets/${dir}/
    sudo chown -R ec2-user ${dir}
    aws s3 sync ${dir}/ s3://msaexp-nirspec/extractions/slitlets/${dir}/ --exclude "*" --include "*phot.*" --include "*raw.*" --acl public-read
done

# Clean up to run again
dirs=`ls | grep v4`
for dir in $dirs; do
    echo $dir
    rm -rf ${dir}/*
    aws s3 rm s3://msaexp-nirspec/extractions/slitlets/${dir}/ --recursive --quiet
done

    """

def get_random_msa_preprocess_file(file=None, **kwargs):
    """
    Get a table row of a file to be processed

    Returns
    -------
    res : None, table row
        Returns a single row from `reprocess_rates` with ``status == 0``, or None if
        nothing found

    """
    from grizli.aws import db

    if file is None:
        WHERE_SELECTION = "status = 0"
    else:
        WHERE_SELECTION = f"rate_file = '{file}'"
        
    rows = db.SQL(f"""SELECT *
        FROM preprocess_nirspec
        WHERE {WHERE_SELECTION} ORDER BY random() limit 1
    """)

    if len(rows) == 0:
        return None
    else:
        return rows


def run_one_msa_preprocess(file=None, sync=True, **kwargs):
    """
    Run a file with status = 0
    """
    import os
    import time
    import boto3

    from grizli.aws import db
    from grizli import utils

    row = get_random_msa_preprocess_file(file=file)

    if row is None:
        print('Nothing to do.')
        with open(os.path.join(HOME, 'nirspec_prep_finished.txt'),'w') as fp:
            fp.write(time.ctime() + '\n')

        return None

    for k in kwargs:
        row[k] = kwargs[k]

    print(f'============  Preprocess NIRSpec  ==============')
    print(f"{row['rate_file'][0]}")
    print(f'========= {time.ctime()} ==========')

    file_prefix = row['rate_file'][0].split('_rate')[0]
    # key = row['root'][0] + '-' + file_prefix
    key = f"{row['root'][0]}-{file_prefix}-{row['rename_f070'][0]}"

    WORKPATH = os.path.join(HOME, key)
    if os.path.exists(WORKPATH):
        print('! already exists, skip')
        return row

    with open(os.path.join(HOME, 'nirspec_prep_history.txt'),'a') as fp:
        fp.write(f"{time.ctime()} {row['rate_file'][0]}\n")

    # Update db
    if sync:
        db.execute(f"""
            UPDATE preprocess_nirspec
            SET status = 1, ctime={time.time()}
            WHERE
                rate_file = '{row['rate_file'][0]}'
                AND root = '{row['root'][0]}'
                AND rename_f070 = {row['rename_f070'][0]}
            """
        )

    #################
    # Run it
    #################
    try:
        status = preprocess_nirspec_msa_file(
            sync=sync, **row[0]
        )
    except:
        status = 5

    if sync:
        # Update db
        db.execute(f"""
            UPDATE preprocess_nirspec
            SET
                status = {status},
                ctime = {time.time()},
                jwst_version = '{jwst.__version__}',
                msaexp_version = '{msaexp.__version__}'
            WHERE
                rate_file = '{row['rate_file'][0]}'
                AND root = '{row['root'][0]}'
                AND rename_f070 = {row['rename_f070'][0]}
            """
        )

    return row


def preprocess_nirspec_msa_file(rate_file='jw01286005001_03101_00002_nrs2_rate.fits', root='jades-gds05-v3', as_fixed=False, rename_f070=False, context='jwst_1225.pmap', sync=True, clean=True, extend_wavelengths=True, undo_flat=True, by_source=False, **kwargs):
    """
    Run preprocessing calibrations for a single NIRSpec exposure
    """
    from grizli import jwst_level1

    os.environ['CRDS_CONTEXT'] = os.environ['CRDS_CTX'] = context
    jwst_utils.set_crds_context()

    # print(rate_file, root)

    outroot = root

    rename_f070 = False

    file_prefix = rate_file.split('_rate')[0]
    # key = root + '-' + file_prefix
    key = f'{root}-{file_prefix}'

    WORKPATH = os.path.join(HOME, key)

    if not os.path.exists(WORKPATH):
        os.makedirs(WORKPATH)

    os.chdir(WORKPATH)

    _ORIG_LOGFILE = utils.LOGFILE
    _NEW_LOGFILE = os.path.join(WORKPATH, file_prefix + '_rate.log.txt')
    utils.LOGFILE = _NEW_LOGFILE

    msg = f"""# {rate_file} {root}
jwst   version = {jwst.__version__}
grizli version = {grizli.__version__}
msaexp version = {msaexp.__version__}
    """
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    # Download to working directory
    mastquery.utils.download_from_mast([rate_file], overwrite=False)

    # os.system(f'aws s3 cp s3://grizli-v2/reprocess_rate/{rate_file} .')

    if not os.path.exists(rate_file):
        msg = f"Failed to download {rate_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return 3

    with pyfits.open(rate_file) as im:
        if 'MSAMETFL' in im[0].header:
            msametf = im[0].header['MSAMETFL']
            mastquery.utils.download_from_mast([msametf], overwrite=False)
            
            msa = msaexp.msa.MSAMetafile(msametf)
            msa.merge_source_ids()
            msa.write(prefix='', overwrite=True)

    use_file = rate_file

    use_prefix = use_file.split('_rate')[0]

    files = [use_file]
    files.sort()

    utils.log_comment(utils.LOGFILE, 'Reset DQ=4 flags', verbose=True)

    for _file in files:
        with pyfits.open(_file, mode='update') as im:
            # print(f'_file unset DQ=4')
            im['DQ'].data -= im['DQ'].data & 4
            im.flush()

    ########################
    # Do the work
    grp = slit_group.NirspecCalibrated(
        rate_file,
        read_slitlet=True,
        make_plot=False,
        area_correction=False,
        prism_threshold=0.999,
        preprocess_kwargs={},
    )
    
    # Above makes {root} _slitlet, _photom and _fs_photom files
    
    # Write individual slitlets
    grp.write_slitlet_files()

    # Global sky
    bkg = grp.get_global_background()
    
    os.system(f'cat {use_prefix}_rate.wave_log.txt >> {_NEW_LOGFILE}')
    
    utils.LOGFILE = _NEW_LOGFILE
    
    # Sync slitlets to S3
    if outroot.split('-')[0] in ['macs0417','macs1423','macs0416','abell370']:
        s3path = 'grizli-canucs/nirspec'
    else:
        s3path = 'msaexp-nirspec/extractions'

    if (outroot not in ['uncover-deep-v1']) & (sync):
        msg = f'Sync slitlets to s3://{s3path}/slitlets/{outroot}/'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        os.system(f'aws s3 sync ./ s3://{s3path}/slitlets/{outroot}/ --exclude "*" --include "*phot.*" --include "*raw.*" --include "*photom.*" --include "{use_prefix}*" --acl public-read --quiet')

    if use_prefix != file_prefix:
        _USE_LOGFILE = os.path.join(WORKPATH, use_prefix + '_rate.log.txt')
        os.system(f"cp {_NEW_LOGFILE} {_USE_LOGFILE}")
    
    if os.path.exists(NIRSPEC_HOME):
        local_path = os.path.join(NIRSPEC_HOME, outroot)
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        msg = f'cp {WORKPATH}/{use_prefix}* {local_path}/'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        os.system(msg)

        msg = f'sudo chown -R ec2-user {local_path}/'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        os.system(msg)

        if msametf is not None:
            msg = f'cp {WORKPATH}/{msametf} {local_path}/'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
            os.system(msg)

    utils.LOGFILE = _ORIG_LOGFILE

    if clean:
        print('Clean up')
        files = glob.glob('*')
        for file in files:
            print(f'rm {file}')
            os.remove(file)

        os.chdir(HOME)
        os.rmdir(WORKPATH)

    return 2

# if __name__ == "__main__":
#     run_one_msa_preprocess()

    
