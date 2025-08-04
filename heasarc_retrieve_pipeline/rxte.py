import glob
import gzip
import os
import shutil
import re
import boto3
import numpy as np
from astropy.io import fits
from astropy.table import Table
from botocore import UNSIGNED
from botocore.config import Config
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash
from datetime import timedelta

DEFAULT_CONFIG = dict(out_data_path="./", input_data_path="./")


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1000),
    task_run_name="rxte_remote_raw_path_{obsid}",
)
def rxte_heasarc_raw_data_path(obsid, cycle=None, prnb=None):
    return os.path.normpath(f"/FTP/rxte/data/archive/AO{cycle}/P{prnb}/{obsid}/")


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1000),
    task_run_name="rxte_base_output_{obsid}",
)
def rxte_base_output_path(config, obsid):
    return os.path.join(config["out_data_path"], obsid)


@task(name="setup_workspace_rxte")
def setup_workspace(raw_data_dir: str, obsid: str):
    logger = get_run_logger()
    l2_dir = os.path.join(raw_data_dir, "l2_files")
    os.makedirs(l2_dir, exist_ok=True)
    paths = {"l2_dir": l2_dir, "raw_data_dir": raw_data_dir, "obsid": obsid}

    logger.info("Searching for PCA Event Mode file (e.g., GX*.evt.gz)...")
    event_patterns = ["GX*.evt.gz", "SE*.evt.gz", "FS*.evt.gz"]
    event_gz_files = []
    for pattern in event_patterns:
        if found_files := glob.glob(os.path.join(raw_data_dir, "**", pattern), recursive=True):
            event_gz_files = found_files
            logger.info(f"Found best available event file: {os.path.basename(event_gz_files[0])}")
            break

    if not event_gz_files:
        logger.warning(
            f"Could not find Event Mode file for OBSID {obsid}. This script can only process Event Mode data.\n"
        )
        return None

    unzipped_event_path = event_gz_files[0][:-3]
    with gzip.open(event_gz_files[0], "rb") as f_in, open(unzipped_event_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    paths["unzipped_event_file"] = unzipped_event_path
    logger.info(f"Unzipped event file: {unzipped_event_path}")
    return paths


@task(name="create_gti_rxte")
def create_gti_with_astropy(paths: dict) -> str:
    # Make a "keep" list of all the good parts.
    logger = get_run_logger()
    raw_data_dir = paths["raw_data_dir"]
    l2_dir = paths["l2_dir"]

    # Find the satellite's filter file
    search_pattern = os.path.join(raw_data_dir, "stdprod", "*.xfl.gz")
    filter_gz_files = glob.glob(search_pattern)
    if not filter_gz_files:
        raise FileNotFoundError(
            f"Standard filter file (*.xfl.gz) not found in {raw_data_dir}/stdprod"
        )
    filter_file_gz = sorted(filter_gz_files)[0]
    filter_file_path = filter_file_gz[:-3]
    with gzip.open(filter_file_gz, "rb") as f_in, open(filter_file_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"Creating GTI from standard filter file: {filter_file_path}")

    # Read the filter data
    with fits.open(filter_file_path) as hdul:
        data_table = Table(hdul[1].data)
        time_res = hdul[1].header.get("TIMEDEL", 16.0)

    mask = (
        (data_table["ELV"] > 10)  # Pointing away from Earth
        & (data_table["OFFSET"] < 0.02)  # Pointing steadily & still
        & (data_table["NUM_PCU_ON"] > 0)  # detectors were on
    )

    # Find all the time intervals that match the above conditions
    good_indices = np.where(mask)[0]
    if len(good_indices) == 0:
        logger.warning("No good time intervals found for the given expression.")
        gti_table = Table(names=("START", "STOP"), dtype=("float64", "float64"))
    else:
        splits = np.where(np.diff(good_indices) > 1)[0]
        starts_idx = np.insert(good_indices[splits + 1], 0, good_indices[0])
        stops_idx = np.append(good_indices[splits], good_indices[-1])
        start_times = data_table["Time"][starts_idx]
        stop_times = data_table["Time"][stops_idx] + time_res
        gti_table = Table([start_times, stop_times], names=("START", "STOP"))

    # Save the list of gti's to a new file in --> GTI file.
    gti_file_path = os.path.join(l2_dir, f"{paths['obsid']}_l2.gti")
    primary_hdu = fits.PrimaryHDU()
    gti_hdu = fits.BinTableHDU(gti_table, name="GTI")
    gti_hdu.header["TELESCOP"] = "XTE"
    hdul = fits.HDUList([primary_hdu, gti_hdu])
    hdul.writeto(gti_file_path, overwrite=True)
    logger.info(f"Successfully created GTI file with Astropy: {gti_file_path}")
    return gti_file_path


@task(name="apply_gti_rxte")
def apply_gti_with_astropy(paths: dict, gti_file: str) -> str | None:
    # Edit using the "keep" list.
    logger = get_run_logger()
    event_file_path = paths["unzipped_event_file"]
    l2_dir = paths["l2_dir"]
    logger.info(f"Applying GTI file {gti_file} to event file {event_file_path}")

    # Open the raw data and the list of gti (GTI file).
    with fits.open(event_file_path) as event_hdul:
        event_table = Table(event_hdul[1].data)
        event_header = event_hdul[1].header
        timezero = event_header.get("TIMEZERO", 0.0)
    with fits.open(gti_file) as gti_hdul:
        gti_table = Table(gti_hdul[1].data)

    # Create a checklist to mark which photons to keep.
    final_mask = np.zeros(len(event_table), dtype=bool)
    event_times_abs = event_table["TIME"] + timezero

    # visit each gti and mark all photons that arrived during that time
    for row in gti_table:
        interval_mask = (event_times_abs >= row["START"]) & (event_times_abs < row["STOP"])
        final_mask |= interval_mask

    # Create a new table with gti
    good_events_table = event_table[final_mask]
    num_good_events = len(good_events_table)
    if num_good_events == 0:
        logger.warning(f"No good events found for OBSID {paths['obsid']} after GTI filtering.")
        return None
    logger.info(f"Found {num_good_events} good events out of {len(event_table)} total.")

    # Save the cl_evt filed.
    cleaned_event_path = os.path.join(l2_dir, f"{paths['obsid']}_cl_evt.fits")
    primary_hdu = fits.PrimaryHDU()
    events_hdu = fits.BinTableHDU(data=good_events_table, header=event_header, name="EVENTS")
    events_hdu.header["HISTORY"] = "Filtered with a custom Python/Astropy routine."
    events_hdu.header["GTI_FILE"] = os.path.basename(gti_file)
    hdul = fits.HDUList([primary_hdu, events_hdu])
    hdul.writeto(cleaned_event_path, overwrite=True)
    return cleaned_event_path


@flow
def process_rxte_obsid(obsid: str, config={}, flags=None, ra: float = None, dec: float = None):
    DEFAULT_CONFIG = dict(out_data_path="./", input_data_path="./")
    current_config = DEFAULT_CONFIG if config is None else config
    logger = get_run_logger()
    logger.info(f"Processing RXTE observation {obsid}")
    raw_data_dir = rxte_base_output_path.fn(config=current_config, obsid=obsid)
    os.makedirs(raw_data_dir, exist_ok=True)

    paths = setup_workspace(raw_data_dir, obsid)

    if paths is None:
        logger.info(f"Pipeline for OBSID {obsid} stopped as no usable data was found.")
        return

    gti_file = create_gti_with_astropy(paths)
    cleaned_event_file = apply_gti_with_astropy(paths, gti_file)

    logger.info("RXTE processing pipeline complete.")
    if cleaned_event_file:
        logger.info(f"Final cleaned event file created at: {cleaned_event_file}")
    else:
        logger.warning("Pipeline finished, but no events remained after cleaning.")