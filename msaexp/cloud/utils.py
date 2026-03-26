import os
import logging

__all__ = [
    "boto3_config_by_bucket",
    "upload_to_s3",
    "download_from_s3",
    "download_file",
    "url_to_s3",
    "s3_to_url",
]

LOGGER = logging.getLogger(__name__)

if len(LOGGER.handlers) == 0:
    ch = logging.StreamHandler()

    ch.setFormatter(
        logging.Formatter(" - %(name)s - %(levelname)s -  %(message)s")
    )

    LOGGER.addHandler(ch)

SCW_S3_KWARGS = dict(
    # service_name='s3',
    region_name="fr-par",
    use_ssl=True,
    endpoint_url="http://s3.fr-par.scw.cloud",
)

CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "fits": "application/fits",
    "csv": "text/csv",
}

SCALEWAY_BUCKETS = ["dja-cloud"]


def boto3_config_by_bucket(
    bucket,
    session_kwargs={},
    env_prefix=None,
    client_kwargs={},
    scaleway=False,
    **kwargs,
):
    """
    Get `boto3` config arguments depending on the bucket name

    Parameters
    ----------
    bucket : str
        Bucket name

    session_kwargs : dict
        Arguments passed to `boto3.Session`

    env_prefix : str
        If provided, set session credentials from environment variables from
        ``aws_access_key_id={env_prefix}AWS_ACCESS_KEY_ID`` and
        ``aws_access_key_id={env_prefix}AWS_SECRET_ACCESS_KEY``.

    client_kwargs : dict
        Arguments passed to ``session.client('s3', **client_kwargs)``.

    scaleway : bool
        Set ``env_prefix='SCW_'`` and ``client_kwargs`` for Scaleway
        S3-compatible object storage.

    Returns
    -------
    session_kwargs, client_kwargs : dict
        Updated arguments depending on the bucket
    """
    # session = boto3.Session(**session_kwargs_)
    if bucket in SCALEWAY_BUCKETS:
        scaleway = True

    client_kwargs_ = client_kwargs.copy()

    if scaleway:
        for k in SCW_S3_KWARGS:
            if k not in client_kwargs_:
                client_kwargs_[k] = SCW_S3_KWARGS[k]

        if (env_prefix is None) & ("profile_name" not in session_kwargs):
            env_prefix = "SCW_"

    session_kwargs_ = session_kwargs.copy()

    if env_prefix is not None:
        has_keys = True
        for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            k_i = f"{env_prefix}{k}"
            has_keys &= os.getenv(k_i) is not None
            session_kwargs_[k.lower()] = os.getenv(k_i)

    return session_kwargs_, client_kwargs_


def upload_to_s3(
    file_name,
    bucket,
    object_name=None,
    object_path="",
    ExtraArgs={},
    content_types=CONTENT_TYPES,
    verbose=True,
    **kwargs,
):
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

    kwargs : dict
        Keyword arguments passed to
        `~msaexp.cloud.utils.boto3_config_by_bucket(bucket, **kwargs)`.

    Returns
    -------
    status : bool
        True if file was uploaded, False if exceptions were raised

    """
    import boto3
    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import ClientError

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.join(object_path, os.path.basename(file_name))

    session_kwargs, client_kwargs = boto3_config_by_bucket(bucket, **kwargs)

    session = boto3.Session(**session_kwargs)

    s3_client = session.client("s3", **client_kwargs)

    # Upload the file
    msg = f"upload {file_name} to s3://{bucket}/{object_name}"

    ExtraArgs["ContentType"] = "text/plain"
    for ext in content_types:
        if object_name.endswith(ext) | object_name.endswith(ext + ".gz"):
            ExtraArgs["ContentType"] = content_types[ext]

    if object_name.endswith(".gz"):
        ExtraArgs["ContentEncoding"] = "gzip"

    try:
        response = s3_client.upload_file(
            file_name, bucket, object_name, ExtraArgs=ExtraArgs
        )

        LOGGER.log(verbose * 10, msg)

        return True

    except ClientError as e:
        LOGGER.error(e)
        LOGGER.error("ClientError " + msg)
        return False

    except S3UploadFailedError as e:
        LOGGER.error(e)
        LOGGER.error("S3UploadFailedError " + msg)
        return False


def url_to_s3(URL, **kwargs):
    """
    Convert HTTP URL to S3

    Parameters
    ----------
    URL : str
        Input URL

    Returns
    -------
    s3_path : str
        S3 path parsed for AWS or Scaleway

    """
    if URL.startswith("s3://"):
        return URL
    if URL.startswith("https://s3.amazonaws.com/"):
        return URL.replace("https://s3.amazonaws.com/", "s3://")
    elif ".scw.cloud/" in URL:
        spl = URL.split("/")
        bucket = spl[2].split(".s3.")[0]
        return "/".join(["s3:/", bucket, *spl[3:]])
    else:
        return URL


def s3_to_url(s3_path, scaleway_region=None, **kwargs):
    """
    Convert S3 path to HTTP URL

    Parameters
    ----------
    s3_path : str
        S3 path parsed for AWS or Scaleway: ``s3://{bucket}/{prefix}``.

    scaleway_region : str, None
        If specified, e.g., 'fr-par', interpret as a Scaleway object
        ``https://{bucket}.s3.{scaleway_region}.scw.cloud/{prefix}``

    Returns
    -------
    URL : str
        HTTP URL

    """
    if s3_path.startswith("http://"):
        return s3_path

    path_split = s3_path.split("s3://")[1].split("/")
    bucket = path_split[0]

    if bucket in SCALEWAY_BUCKETS:
        scaleway_region = "fr-par"

    prefix = "/".join(path_split[1:])

    if scaleway_region is not None:
        URL = f"https://{bucket}.s3.{scaleway_region}.scw.cloud/{prefix}"
    else:
        URL = f"https://s3.amazonaws.com/{bucket}/{prefix}"

    return URL


def download_file(
    URL, output_path="./", overwrite=False, check_s3=True, **kwargs
):
    """
    Download a remote file from HTTP or S3

    Parameters
    ----------
    URL : str
        HTTP or S3 URL

    output_path : str
        Local path of downloaded file

    overwrite : bool
        Overwrite local file if already exists

    check_s3 : bool
        If URL starts with "https://s3.amazonaws.com/", try to first download
        with the `boto3` S3 API.

    Returns
    -------
    local_file : str, None
        Path to downloaded file

    """
    import requests

    if "force" in kwargs:
        overwrite = kwargs["force"]

    local_file = os.path.join(output_path, os.path.basename(URL))

    if os.path.exists(local_file) & (not overwrite):
        return local_file

    s3_path = url_to_s3(URL)

    if check_s3 & s3_path.startswith("s3://"):
        s3_file = download_from_s3(s3_path, output_path=output_path, **kwargs)

        if s3_file is None:
            if URL.startswith("s3://"):
                return None

            msg = f"download_from_s3('{s3_path}') failed, fall back to http"
            LOGGER.debug(msg)
        else:
            return s3_file

    msg = f"Download {URL} to {local_file}"
    LOGGER.debug(msg)

    resp = requests.get(URL.replace("+", "%2B"))

    # save to file
    with open(local_file, "wb") as FLE:
        FLE.write(resp.content)

    return local_file


def download_from_s3(
    path="s3://grizli-v2/scratch/junk.txt",
    output_path="./",
    ExtraArgs={"RequestPayer": "requester"},
    overwrite=True,
    verbose=True,
    **kwargs,
):
    """
    Download file from S3 object storage

    Parameters
    ----------
    path : str
        Full S3 object path ``s3://[bucket]/[prefix]/[filename]``

    output_path : str
        Local directory for output, i.e., ``[output_path]/[filename]``.

    ExtraArgs : dict
        Arguments passed to
        ``boto3.Session.Resource('s3').Bucket(bucket).download_file``.

    overwrite : bool
        Overwrite local file if it exists

    kwargs : dict
        Keyword arguments passed to
        `~msaexp.cloud.utils.boto3_config_by_bucket(bucket, **kwargs)`.

    Returns
    -------
    local_file : str, None
        Path to local file if found, None if download failed

    """
    import boto3
    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import ClientError

    path_split = path.split("s3://")[1].split("/")
    bucket = path_split[0]

    session_kwargs, client_kwargs = boto3_config_by_bucket(bucket, **kwargs)

    session = boto3.Session(**session_kwargs)
    s3 = session.resource("s3", **client_kwargs)

    # s3 = boto3.resource('s3')

    bkt = s3.Bucket(bucket)
    file_prefix = "/".join(path_split[1:])

    local_file = os.path.join(output_path, os.path.basename(file_prefix))

    if os.path.exists(local_file) & (not overwrite):
        if verbose:
            print(f"{local_file} exists")

        return local_file

    msg = f"Download {path} to {local_file}"

    try:
        bkt.download_file(file_prefix, local_file, ExtraArgs=ExtraArgs)
        LOGGER.log(verbose * 10, msg)
        return local_file

    except ClientError as e:
        LOGGER.error(e)
        LOGGER.error("ClientError " + msg)
        return None

    except S3TransferFailedError as e:
        LOGGER.error(e)
        LOGGER.error("S3TransferFailedError " + msg)
        return None
