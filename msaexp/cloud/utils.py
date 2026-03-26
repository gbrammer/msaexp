import os
import logging

__all__ = [
    "upload_to_s3"
]

LOGGER = logging.getLogger(__name__)

if len(LOGGER.handlers) == 0:
    ch = logging.StreamHandler()

    ch.setFormatter(
        logging.Formatter(" - %(name)s - %(levelname)s -  %(message)s"))

    LOGGER.addHandler(ch)

SCW_S3_KWARGS = dict(
    # service_name='s3',
    region_name='fr-par',
    use_ssl=True,
    endpoint_url='http://s3.fr-par.scw.cloud',
)

CONTENT_TYPES = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'gif': 'image/gif',
    'fits': 'application/fits',
    'csv': 'text/csv',
}

def upload_to_s3(file_name, bucket, object_name=None, object_path='', ExtraArgs={}, content_types=CONTENT_TYPES, verbose=True, session_kwargs={}, env_prefix=None, client_kwargs={}, scaleway=False):
    """
    Upload a file to an S3 bucket

    Parameters
    ----------
    file_name : str
        Path to local file to upload

    bucket : str
        Bucket name

    object_name, object_path : str, None
        S3 object name. If not specified then
        ``os.path.join(object_path, os.path.basename(file_name)`` is used.

    ExtraArgs : dict
        Passed to ``s3_client.upload_file``, e.g., 
        ``ExtraArgs={'ACL':'public-read'}``.  Content 

    content_types : dict
        Mapping between file extensions and content types to attach to
        ``ExtraArgs['ContentType']``.  Defaults to 'text/plain' if the
        extension of ``file_name`` not found.

    session_kwargs : dict
        Arguments passed to `boto3.Session`

    env_prefix : str
        If provided, set session credentials from environment variables from
        ``aws_access_key_id=[env_prefix]AWS_ACCESS_KEY_ID`` and
        ``aws_access_key_id=[env_prefix]AWS_SECRET_ACCESS_KEY``.

    client_kwargs : dict
        Arguments passed to ``session.client('s3', **client_kwargs)``.

    scaleway : bool
        Set ``env_prefix='SCW_'`` and ``client_kwargs`` for Scaleway
        S3-compatible object storage.

    Returns
    -------
    status : bool
        True if file was uploaded, False if exceptions were raised

    """
    import boto3
    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import ClientError

    import os

    if scaleway:
        client_kwargs = SCW_S3_KWARGS
        if (env_prefix is None) & ('profile_name' not in session_kwargs):
            env_prefix = 'SCW_'
            
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.join(object_path, os.path.basename(file_name))

    session_kwargs_ = session_kwargs.copy()

    if env_prefix is not None:
        has_keys = True
        for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            k_i = f"{env_prefix}{k}"
            has_keys &= os.getenv(k_i) is not None
            session_kwargs_[k.lower()] = os.getenv(k_i)

    session = boto3.Session(**session_kwargs_)

    s3_client = session.client(
        's3',
        **client_kwargs
    )

    # Upload the file
    msg = f'upload {file_name} to s3://{bucket}/{object_name}'

    ExtraArgs['ContentType'] = 'text/plain'
    for ext in content_types:
        if object_name.endswith(ext) | object_name.endswith(ext + '.gz'):
            ExtraArgs['ContentType'] = content_types[ext]

    if object_name.endswith('.gz'):
        ExtraArgs['ContentEncoding'] = "gzip"

    try:
        response = s3_client.upload_file(
            file_name,
            bucket,
            object_name,
            ExtraArgs=ExtraArgs
        )

    except ClientError as e:
        LOGGER.error(e)
        LOGGER.error('ClientError ' + msg)
        return False

    except S3UploadFailedError as e:
        LOGGER.error(e)
        LOGGER.error('S3UploadFailedError ' + msg)
        return False

    LOGGER.log(verbose * 10, msg)

    return True
