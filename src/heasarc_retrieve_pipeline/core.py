import os
import re
import shutil
import sys
import glob
import traceback
import pytest
import warnings
import typing
from astropy.table import hstack, Table
from astroquery.heasarc import Heasarc
from astropy.coordinates import SkyCoord


from .nustar import nu_heasarc_raw_data_path as nu_raw_data_path
from .nustar import process_nustar_obsid
from .nicer import ni_raw_data_path, process_nicer_obsid
from .rxte import rxte_heasarc_raw_data_path as rxte_raw_data_path
from .rxte import process_rxte_obsid

from prefect import flow, task, get_run_logger


def _download_pysmartdl(url: str, dest: str):
    from pySmartDL import SmartDL

    obj = SmartDL(url, dest)
    obj.start()
    return obj.get_dest()


def remote_data_url(mission, obsid, time, cycle=None, prnb=None):
    url = (
        "https://heasarc.gsfc.nasa.gov/"
        + MISSION_CONFIG[mission]["path_func"](obsid, time, cycle, prnb)
        + "/"
    )
    return url


def download_cmd(url: str, dest: str):
    try:
        return _download_pysmartdl(url, dest), None
    except Exception as e:
        return None, str(e)


@task(task_run_name="get_remote_directory_listing_{url}")
def get_remote_directory_listing(url: str):
    """Give the list of files in the remote directory."""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    from bs4 import BeautifulSoup

    url = url.replace(" ", "%20")
    req = Request(url)
    try:
        a = urlopen(req).read()
    except HTTPError:
        return None

    soup = BeautifulSoup(a, "html.parser")
    x = soup.find_all("a")
    urls = []
    for i in x:
        file_name = i.extract().get_text()
        url_new = url + file_name
        url_new = url_new.replace(" ", "%20")
        if file_name[-1] == "/" and file_name[0] != ".":
            urls.append(url_new)
            url_new = get_remote_directory_listing.fn(url_new)
            if url_new is None:
                continue
            urls.extend(url_new)
        else:
            urls.append(url_new)

    return urls


@task(task_run_name="download_{node}")
def download_node(
    node: str,
    base_url: str,
    outdir: str,
    cut_ndirs: int = 0,
    test_str: str = ".",
    test: bool = False,
):
    logger = get_run_logger()
    local_ver = os.path.join(outdir, *node.replace(base_url, "").split("/")[cut_ndirs:])
    if test_str is not None and test_str not in local_ver:
        logger.debug(f"Ignoring {node}")
        return None
    logger.info(f"Downloading {node} to {local_ver}")
    if os.path.exists(local_ver):
        logger.info(f"{local_ver} exists")
        return None

    is_dir = local_ver.endswith("/")

    if is_dir:
        if not test:
            os.makedirs(local_ver, exist_ok=True)
        else:
            logger.info(f"Faked creation of {local_ver}")
        return local_ver

    if not test:
        fname, exc_string = download_cmd(node, local_ver)
    else:
        logger.info(f"Faked download of {node} to {local_ver}")
        fname = local_ver
    if fname is None:
        logger.warning(f"Error downloading {node}: {exc_string}")

    return local_ver


@task(task_run_name="recursive_download_s3_{url}")
def recursive_download_s3(
    url: str,
    outdir: str,
    cut_ndirs: int = 0,
    test_str: str = ".",
    test: bool = False,
    re_include: str = "",
    re_exclude: str = "",
):
    import boto3
    import botocore
    from urllib.parse import urlparse

    os.makedirs(outdir, exist_ok=True)
    logger = get_run_logger()
    logger.info("Recursively downloading from S3...")

    # Parse S3 URL
    parsed = urlparse(url)
    bucket_name = parsed.netloc

    # Adapted from astroquery.heasarc
    logger.info("Enabling anonymous cloud data access ...")
    config = botocore.client.Config(signature_version=botocore.UNSIGNED)
    s3_resource = boto3.resource("s3", config=config)

    s3_client = s3_resource.meta.client

    path = url.replace(f"s3://{bucket_name}/", "")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)

    re_include = re.compile(re_include) if re_include != "" else None
    re_exclude = re.compile(re_exclude) if re_exclude != "" else None

    content = response.get("Contents", [])
    local_vers = []
    for obj in content:
        key = obj["Key"]

        if re_include is not None and not re_include.search(key):
            logger.info(f"Skipping {key} because not included in {re_include.pattern}")
            continue
        if re_exclude is not None and re_exclude.search(key):
            logger.info(f"Skipping {key} because excluded in {re_exclude.pattern}")
            continue

        path2 = "/".join(path.strip("/").split("/")[:-1])
        dest = os.path.join(outdir, key[len(path2) + 1 :])
        if test_str is not None and test_str not in dest:
            logger.debug(f"Ignoring {key}")
            continue
        if os.path.exists(dest):
            logger.info(f"{dest} already exists, skipping download.")
            local_vers.append(dest)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        logger.info(f"Downloading s3://{bucket_name}/{key} to {dest}")

        if not test:
            s3_client.download_file(bucket_name, key, dest)
        else:
            logger.info(f"Faked download of s3://{bucket_name}/{key} to {dest}")
        local_vers.append(dest)
    return local_vers


@flow(flow_run_name="recursive_download_https_{url}")
def recursive_download_https(
    url: str,
    outdir: str,
    cut_ndirs: int = 0,
    test_str: str = ".",
    test: bool = False,
    re_include: str = "",
    re_exclude: str = "",
):
    re_include = re.compile(re_include) if re_include != "" else None
    re_exclude = re.compile(re_exclude) if re_exclude != "" else None

    # rec_down_file = os.path.join(outdir, "DOWNLOAD_DONE.txt")
    # if os.path.exists(rec_down_file):
    #     logger.info(f"Download already done for {url}")
    #     return
    logger = get_run_logger()
    logger.info("Getting remote directory listing...")
    listing = get_remote_directory_listing.fn(url)
    if listing is None or listing == []:
        logger.warning(f"No data found in remote directory {url}")
        return False

    base_url = "/".join(url.rstrip("/").split("/")[:-1])
    local_vers = []
    os.makedirs(outdir, exist_ok=True)
    for node in listing:
        if re_include is not None and not re_include.search(node):
            logger.info(f"Skipping {node} because not included in {re_include.pattern}")
            continue
        if re_exclude is not None and re_exclude.search(node):
            logger.info(f"Skipping {node} because excluded in {re_exclude.pattern}")
            continue
        local_vers.append(
            download_node(
                node,
                base_url,
                outdir,
                cut_ndirs=cut_ndirs,
                test_str=test_str,
                test=test,
            )
        )
    # open(rec_down_file, "a").close()
    return local_vers


@task(task_run_name="copy_local_directory_{url}")
def copy_local_directory(url: str, outdir: str):
    """Copy a local directory to the output directory."""
    outpath = os.path.join(outdir, url.rstrip("/").split("/")[-1])
    logger = get_run_logger()
    logger.info(f"Copying local directory {url} to {outpath}")

    shutil.copytree(url.rstrip("/"), outpath, dirs_exist_ok=True)

    return os.walk(outpath)


@flow(flow_run_name="recursive_download_{url}")
def recursive_download(
    url: str,
    outdir: str,
    cut_ndirs: int = 0,
    test_str: str = ".",
    test: bool = False,
    re_include: str = "",
    re_exclude: str = "",
):

    if url.startswith("http"):
        return recursive_download_https(
            url, outdir, cut_ndirs, test_str, test, re_include, re_exclude
        )

    if url.startswith("s3://"):
        return recursive_download_s3(url, outdir, cut_ndirs, test_str, test, re_include, re_exclude)

    return copy_local_directory(url, outdir)  # For local directories, we just copy them directly


MISSION_CONFIG = {
    "nustar": {
        "table": "numaster",
        "path_func": nu_raw_data_path,
        "expo_column": "exposure_a",
        "additional": "solar_activity",
        "obsid_processing": process_nustar_obsid,
        "name_column": "name",
    },
    "nicer": {
        "table": "nicermastr",
        "path_func": ni_raw_data_path,
        "expo_column": "exposure",
        "additional": "",
        "obsid_processing": process_nicer_obsid,
        "name_column": "name",
    },
    "rxte": {
        "table": "xtemaster",
        "path_func": rxte_raw_data_path,
        "expo_column": "exposure",
        "additional": "cycle, prnb",
        "obsid_processing": process_rxte_obsid,
        "name_column": "target_name",
    },

}


@task(task_run_name="read_config_{config_file}")
def read_config(config_file: str):
    import yaml

    with open(config_file, "r") as f:
        config = yaml.load(f)
    return config

@task(log_prints=True)
def retrieve_heasarc_table_by_position(
    ra_deg: float, dec_deg: float, mission: str = "nustar", radius_deg: float = 0.1
):
    logger = get_run_logger()
    logger.info(
        f"Retrieving HEASARC table for {mission} at RA: {ra_deg}, Dec: {dec_deg}, Radius: {radius_deg}"
    )

    expo_name = MISSION_CONFIG[mission]["expo_column"]
    additional = MISSION_CONFIG[mission]["additional"]
    table = MISSION_CONFIG[mission]["table"]
    source_name_col = MISSION_CONFIG[mission]["name_column"]

    select_columns = f"{source_name_col} as source_name, obsid, time, {expo_name}, ra, dec, __row"

    if mission != "rxte":
        select_columns += ", public_date"

    if additional:
        select_columns += f", {additional}"

    query = f"""SELECT {select_columns}
        FROM public.{table} as cat
        where
        contains(point('ICRS',cat.ra,cat.dec),circle('ICRS',{ra_deg},{dec_deg},{radius_deg}))=1
        and
        cat.{expo_name} >= 0 order by cat.time
        """

    results = Heasarc.query_tap(query).to_table()

    return results


@task(task_run_name="retrieve_info_for_obsid_{obsid}")
def retrieve_info_for_obsid(obsid, mission: str = "nustar"):
    """
    Retrieve the observation information for a given obsid from the HEASARC table.
    """
    expo_name = MISSION_CONFIG[mission]["expo_column"]
    additional = MISSION_CONFIG[mission]["additional"]
    table = MISSION_CONFIG[mission]["table"]
    name_column = MISSION_CONFIG[mission]["name_column"]
    if additional != "":
        additional = f", {additional}"
    query = f"""SELECT {name_column}, cycle, obsid, time, {expo_name}, ra, dec, __row {additional}
        FROM public.{table} as cat
        where
        cat.obsid='{obsid}'
        and
        cat.{expo_name} >= 0 order by cat.time
        """

    results = Heasarc.query_tap(query).to_table()
    return results

 
@task
def get_source_position(source: str):

    pos = SkyCoord.from_name(f"{source}")

    return pos


@task
def retrieve_heasarc_table_by_source_name(
    source: str, mission: str = "nustar", radius_deg: float = 0.1
):
    pos = get_source_position.fn(source)
    results = retrieve_heasarc_table_by_position.fn(
        pos.ra.deg, pos.dec.deg, mission=mission, radius_deg=radius_deg
    )
    return results


@task
def retrieve_heasarc_data_by_source_name_old(
    source: str,
    outdir: str = "out",
    mission: str = "nustar",
    radius_deg: float = 0.1,
    test: bool = False,
):

    logger = get_run_logger()
    pos = get_source_position(source)

    results = retrieve_heasarc_table_by_position.fn(
        pos.ra.deg, pos.dec.deg, mission=mission, radius_deg=radius_deg
    )

    for row in results:
        logger.info(f"{row['obsid']}, {row['time']}")
    cwd = os.getcwd()
    for obsid, time, cycle, prnb in zip(results["obsid"], results["time"], results["cycle"], results["prnb"]):
        os.chdir(cwd)
        url = remote_data_url(mission, obsid, time, cycle, prnb)
        recursive_download(
            url,
            outdir,
            cut_ndirs=0,
            test_str=".",
            test=test,
            wait_for=[remote_data_url],
        )
        if test:
            break
        os.chdir(outdir)
        process_nustar_obsid(
            obsid,
            config=None,
            ra=pos.ra.deg,
            dec=pos.dec.deg,
            wait_for=[recursive_download],
            return_state=True,
        )

    return results

@flow
def retrieve_and_process_data(
    result_table: Table,
    source_position: typing.Union[SkyCoord, None] = None,
    mission: str = "nustar",
    outdir: str = "out",
    test: bool = False,
    flags ={},
    force_heasarc: bool = False,
    force_s3: bool = False,
):

    cwd = os.getcwd()
    try:
        processing = MISSION_CONFIG[mission]["obsid_processing"]
        links = Heasarc.locate_data(result_table, catalog_name=MISSION_CONFIG[mission]["table"])
        if force_s3:
            link_col_name = "aws"
        elif force_heasarc:
            link_col_name = "access_url"
        elif "SCISERVER_USER_ID" in os.environ:
            link_col_name = "sciserver"
        else:
            link_col_name = "aws"

        for i, row in enumerate(result_table):
            obsid = row["obsid"]
            if source_position is not None:
                ra = source_position.ra.deg
                dec = source_position.dec.deg
            else:
                ra = row["ra"]
                dec = row["dec"]

            os.chdir(cwd) # This line can stay
            recursive_download(links[i][link_col_name], outdir, test_str=".", test=test)
            if test:
                break

            os.chdir(outdir) # This is the line that causes the issue

            processing(
                obsid,
                config=None,
                ra=ra,
                dec=dec,
                flags=flags,
                wait_for=[recursive_download],
                return_state=True,
            )
    finally:
        os.chdir(cwd)

    return result_table

@flow
def retrieve_heasarc_data_by_source_name(
    source: str,
    outdir: str = "out",
    mission: str = "nustar",
    radius_deg: float = 0.1,
    test: bool = False,
    force_heasarc: bool = False,
    force_s3: bool = False,
):

    logger = get_run_logger()
    pos = get_source_position(source)

    results = retrieve_heasarc_table_by_position.fn(
        pos.ra.deg, pos.dec.deg, mission=mission, radius_deg=radius_deg
    )
    try:
        links = Heasarc.locate_data(results, catalog_name=MISSION_CONFIG[mission]["table"])
    except Exception as e:
        logger.error(f"Error using astroquery to locate data: {str(e)}")
        return retrieve_heasarc_data_by_source_name_old.fn(
            source=source,
            outdir=outdir,
            mission=mission,
            radius_deg=radius_deg,
            test=test,
        )
    results = retrieve_and_process_data(
        result_table=results,
        source_position=pos,
        mission=mission,
        outdir=outdir,
        test=test,
        force_heasarc=force_heasarc,
        force_s3=force_s3,
    )

    return results


@flow
def retrieve_heasarc_data_by_obsid(
    obsid: str,
    outdir: str = "out",
    mission: str = "nustar",
    test: bool = False,
    flags: dict = {},
    force_heasarc: bool = False,
    force_s3: bool = False,
):

    logger = get_run_logger()

    results = retrieve_info_for_obsid(obsid, mission=mission)
    if not results:
        logger.warning(f"No observations found for OBSID {obsid} in HEASARC query.")
        return None   

    results = retrieve_and_process_data(
        result_table=results,
        source_position=None,
        mission=mission,
        outdir=outdir,
        test=test,
        flags=flags,
        force_heasarc=force_heasarc,
        force_s3=force_s3,
        wait_for=[retrieve_info_for_obsid],
    )
    return results

