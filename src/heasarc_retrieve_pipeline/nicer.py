
import re

from astropy.time import Time

import glob

from datetime import timedelta

from prefect import flow, task, get_run_logger

from prefect.tasks import task_input_hash
import subprocess
import os

from .barycenter import barycenter_file

try:
    HAS_HEASOFT = True
    import heasoftpy as hsp
except ImportError:
    HAS_HEASOFT = False
    print("Warning: heasoftpy not installed. NICER L2 pipeline functionality will be disabled.")


DEFAULT_CONFIG = dict(out_data_path="./", input_data_path="./")


valid_re = re.compile(r"ni[0-9]{11}")


@task
def ni_raw_data_path(obsid, time, **kwargs):
    from astropy.time import Time

    mjd = Time(time.data, format="mjd")
    mjd_dt = mjd.to_datetime()

    return os.path.normpath(f"/FTP/nicer/data/obs/{mjd_dt.year}_{mjd_dt.month:02d}//{obsid}/")


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1000),
    task_run_name="ni_base_output_{obsid}",
)
def ni_base_output_path(config, obsid):
    return os.path.join(config["out_data_path"], obsid)


@task
def ni_pipeline_output_path(config, obsid):
    return os.path.join(config["out_data_path"], obsid + "/l2files/")


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1000),
    task_run_name="ni_pipeline_done_file_{obsid}",
)
def ni_pipeline_done_file(obsid, config):
    return os.path.join(ni_pipeline_output_path.fn(obsid, config), "PIPELINE_DONE.TXT")


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1000),
    task_run_name="nicerl2_{obsid}",
)
def ni_run_l2_pipeline(obsid, config, flags=None):

    logger = get_run_logger()
    if not HAS_HEASOFT:
        logger.error("heasoftpy not installed, cannot run NICER L2 pipeline.")
        raise ImportError("heasoftpy not installed")

    ev_dir = ni_pipeline_output_path.fn(config=config, obsid=obsid)
    os.makedirs(ev_dir, exist_ok=True)

    full_pipe_done_file_path = os.path.join(ev_dir, "PIPELINE_DONE.TXT")

    if os.path.exists(full_pipe_done_file_path):
        logger.info(
            f"Data for {obsid} already preprocessed. Done file found at: {full_pipe_done_file_path}"
        )
        return ev_dir

    nicerl2_hsp_task = hsp.HSPTask("nicerl2")
    logger.info("Running Nicer L2 pipeline for OBSID: %s", obsid)

    datadir = ni_base_output_path.fn(config=config, obsid=obsid)
    os.makedirs(ev_dir, exist_ok=True)
    logger.info(f"Ensuring desired final output directory exists: {ev_dir}")

    params = {
        "indir": datadir,
        "cldir": ev_dir,
        "clobber": True,
        "chatter": 5,
        "geomag_path": "/FTP/caldb/data/gen/pcf/geomag/",
        
    }
    if flags:
        params.update(flags)
    command = ["nicerl2"]
    for key, value in params.items():
        command.append(f"{key}={value}")

    log_file_path = os.path.join(ev_dir, f"nicerl2_process_{obsid}.log")
    error_log_path = os.path.join(ev_dir, f"nicerl2_process_{obsid}.err")

    logger.info(f"Executing command: {' '.join(command)}")
    try:
        with open(log_file_path, "w") as log_f, open(error_log_path, "w") as err_f:
            result = subprocess.run(command, stdout=log_f, stderr=err_f, check=False)
        if result.returncode != 0:
            with open(error_log_path, "r") as f:
                error_output = f.read()
            logger.error(f"nicerl2 FAILED! Error:\n{error_output}")
            raise RuntimeError(f"nicerl2 failed. See log: {error_log_path}")
    except FileNotFoundError:
        logger.error("FATAL: 'nicerl2' command not found.")
        raise

    logger.info(f"nicerl2 completed successfully for OBSID: {obsid}")
    with open(full_pipe_done_file_path, "w") as f:
        f.write(f"Completed on {Time.now().iso}")

    return ev_dir


@flow(flow_run_name="ni_barycenter_{obsid}")
def barycenter_data(obsid: str, ra: float, dec: float, config: dict):
    logger = get_run_logger()
    outdir = ni_base_output_path.fn(config=config, obsid=obsid)
    logger.info(f"Barycentering NICER data in directory {outdir}")
    infile = os.path.join(outdir, "l2files", f"ni{obsid}_0mpu7_cl.evt")
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Event file not found: {infile} delete the data and retry :)")

    orbit_file = os.path.join(outdir, "auxil", f"ni{obsid}.orb.gz")
    if not os.path.exists(orbit_file):
        raise FileNotFoundError(f"Orbit file not found: {orbit_file}")

    return barycenter_file(
        infile=infile,
        attorb=orbit_file,
        ra=float(ra),
        dec=float(dec),
    )


@flow
def process_nicer_obsid(obsid: str, config={}, ra="NONE", dec="NONE", flags=None):
    current_config = DEFAULT_CONFIG if config is None else config
    logger = get_run_logger()
    logger.info(f"Processing Nicer observation {obsid}")
    base_output_dir_for_obsid = ni_base_output_path.fn(config=current_config, obsid=obsid)
    os.makedirs(base_output_dir_for_obsid, exist_ok=True)
    logger.info(f"Ensured base output directory exists: {base_output_dir_for_obsid}")
    ni_run_l2_pipeline(obsid, config=current_config, flags=flags)
    barycenter_data(obsid, ra=ra, dec=dec, config=current_config)
    logger.info(f"Finished processing Nicer observation {obsid}")