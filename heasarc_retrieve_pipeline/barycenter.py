import os
from datetime import timedelta
from prefect import task, get_run_logger
from prefect.tasks import task_input_hash

try:
    import heasoftpy as hsp

    HAS_HEASOFT = True
except ImportError:
    HAS_HEASOFT = False


@task(
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=90),
    task_run_name="barycenter_{infile}",
)
def barycenter_file(infile: str, attorb: str, ra, dec):
    if not HAS_HEASOFT:
        raise ImportError("heasoftpy is required for barycenter correction but is not installed.")

    logger = get_run_logger()
    logger.info(f"Barycentering {infile}")

    outfile = infile.replace(".evt", "_bary.evt")
    logger.info(f"Output file: {outfile}")

    hsp.barycorr(
        infile=infile,
        outfile=outfile,
        ra=ra,
        dec=dec,
        ephem="JPLEPH.430",
        refframe="ICRS",
        clobber="yes",
        orbitfiles=attorb,
        chatter=5,
    )

    if not os.path.exists(outfile):
        raise FileNotFoundError(f"Barycentered output file not created: {outfile}")

    return outfile