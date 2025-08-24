import panel as pn
import asyncio
import pandas as pd
import os
import logging
import re
import shutil
import glob
import json
import h5py  
import atexit
import subprocess
from astroquery.heasarc import Heasarc
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates.name_resolve import NameResolveError

from src.heasarc_retrieve_pipeline.core import retrieve_heasarc_data_by_obsid
from src.dashboard.HDF5.h5 import process_observation, create_h5_generator_tab

TEST_MODE = True
prefect_process = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_prefect_server():
    """Starts the Prefect server as a background process."""
    global prefect_process
    try:
        logger.info("Starting Prefect server in the background...")
        # Use Popen to run the server as a non-blocking background process
        prefect_process = subprocess.Popen(["prefect", "server", "start"])
        logger.info(f"Prefect server started with process ID: {prefect_process.pid}")
    except FileNotFoundError:
        logger.error("'prefect' command not found. Make sure Prefect is installed and in your PATH.")
    except Exception as e:
        logger.error(f"Failed to start Prefect server: {e}", exc_info=True)

def stop_prefect_server():
    """Stops the background Prefect server process if it's running."""
    global prefect_process
    if prefect_process:
        logger.info(f"Shutting down Prefect server (PID: {prefect_process.pid})...")
        prefect_process.terminate() # Send a termination signal
        prefect_process.wait()      # Wait for the process to exit
        logger.info("Prefect server shut down.")

# Register the cleanup function to run when the script exits
atexit.register(stop_prefect_server)

# Start the server when the script is first loaded
start_prefect_server()

MISSION_INFO = {
    "nustar": {
        "table": "numaster",
               "expo_column": "exposure_a",
               "stop_col": "end_time"
    },
    "nicer": {
        "table": "nicermastr",
              "expo_column": "exposure",
              "stop_col": "end_time"
    },
    "rxte": {
        "table": "xtemaster",
             "expo_column": "exposure",
             "stop_col": "stop_time"
    }, 
}
def partition_obsids_by_exposure(obsids: list, mission: str):
    """
    Queries HEASARC to partition a list of OBSIDs into two groups based on exposure time.

    Separates OBSIDs with a recorded exposure time greater than zero from those
    with zero or no recorded exposure time. This is useful for deciding which
    processing flags to use for NICER data.

    Args:
        obsids (list): A list of observation IDs to check.
        mission (str): The name of the mission (e.g., 'nicer', 'nustar').

    Returns:
        tuple: A tuple containing two lists:
        - The first list has OBSIDs with non-zero exposure.
        - The second list has OBSIDs with zero or unknown exposure.
    """

    if not obsids:
        return [], []
    try:
        table_name = MISSION_INFO[mission]["table"]
        expo_col = MISSION_INFO[mission]["expo_column"]
        obsid_list_str = "','".join(obsids)
        query = f"SELECT obsid, {expo_col} FROM public.{table_name} WHERE obsid IN ('{obsid_list_str}')"
        results_table = Heasarc.query_tap(query).to_table()
        if len(results_table) == 0:
            return [], obsids
        df = results_table.to_pandas()
        valid_df = df[df[expo_col] > 0]
        nonzero_expo_obsids = valid_df['obsid'].astype(str).tolist()
        zero_expo_obsids = list(set(obsids) - set(nonzero_expo_obsids))
        return nonzero_expo_obsids, zero_expo_obsids
    except Exception as e:
        logger.error(f"Failed to partition OBSIDs by exposure: {e}. Assuming all have valid exposure.")
        return obsids, []

def retrieve_heasarc_data_by_obsid_wrapper(source: str, obsid: str, base_outdir: str, mission: str, flags: dict):
    """
    Downloads and processes a single observation from HEASARC if not already done.

    This function acts as a wrapper around the core `retrieve_heasarc_data_by_obsid`
    function. It first checks for a "PIPELINE_DONE.TXT" file to avoid
    re-downloading and re-processing data that already exists.

    Args:
        source (str): The name of the astronomical source.
        obsid (str): The specific Observation ID to process.
        base_outdir (str): The base directory where the output for the source will be stored.
        mission (str): The name of the mission (e.g., 'nicer').
        flags (dict): A dictionary of custom flags for the data processing pipeline.

    Returns:
        str: The absolute path to the directory containing the processed data for the obsid.
             Returns the path without running the pipeline if the data already exists.

    Side effects:
        - Creates directories on the file system for the observation data.
        - Downloads a large amount of data from HEASARC.
        - Runs an external data processing pipeline.
        - Creates a "obsid_done.TXT" file upon completion.
    """
    obs_dir = os.path.join(base_outdir, obsid)
    abs_obs_dir = os.path.abspath(obs_dir)
    if mission == 'nicer':
     bary_file_pattern = os.path.join(abs_obs_dir, '**', '*_cl_bary.evt')
     found_files = glob.glob(bary_file_pattern, recursive=True)
    elif mission == 'nustar':
     bary_file_pattern = os.path.join(abs_obs_dir, '**', '*_src1_bary.evt')   
     found_files = glob.glob(bary_file_pattern, recursive=True)
    elif mission == 'rxte':
     bary_file_pattern = os.path.join(abs_obs_dir, '**', '*_cl_evt.fits')   
     found_files = glob.glob(bary_file_pattern, recursive=True)
    
    if found_files:
        logger.info(f"Barycenter-corrected file found for {obsid}. Skipping pipeline.")
        return abs_obs_dir
    
    logger.info(f"Starting REAL processing for OBSID: {obsid} with flags: {flags}")
    use_s3 = True
    if mission == 'rxte':
      logger.warning("Forcing HTTPS download for RXTE due to potential S3 path issues.")
      use_s3 = False

    retrieve_heasarc_data_by_obsid(
     obsid=str(obsid),
     outdir=os.path.abspath(base_outdir),
     mission=mission,
     test=False,
     flags=flags,
     force_s3=use_s3,
     force_heasarc=(not use_s3)
   )

    os.makedirs(abs_obs_dir, exist_ok=True)
    with open(os.path.join(abs_obs_dir, "obsid_done.TXT"), "w") as f:
        f.write(f"Completed with flags: {flags}")

    logger.info(f"Successfully processed and downloaded OBSID: {obsid}")
    return abs_obs_dir

def get_obsids_from_heasarc_direct(source_name: str, mission: str):

        heasarc = Heasarc()
        mission_info = MISSION_INFO[mission]
        table_name = mission_info["table"]
        
        position = SkyCoord.from_name(source_name)
        result_table = heasarc.query_region(position, catalog=table_name, radius='90 arcmin')
        if result_table is None: return pd.DataFrame()
        
        df = result_table.to_pandas()

        # 1. Normalize all column names to lowercase to prevent case-sensitivity issues.
        df.columns = [col.lower() for col in df.columns]
        
        # 2. Determine the name of the stop time column we will create or use.
        # For RXTE, we calculate 'stop_time'; for others, we look for 'end_time'.
        stop_time_column = 'stop_time' if mission == 'rxte' else 'end_time'

        # 3. Check for the column's existence and calculate it if it's missing.
        if stop_time_column not in df.columns:
            # This block now handles all cases where the precise stop time is missing.
            
            if mission == 'rxte' and 'duration' in df.columns:
                # Precise calculation for RXTE
                logger.info("RXTE: Calculating precise stop time from 'duration'.")
                df[stop_time_column] = df['time'] + (pd.to_numeric(df['duration'], errors='coerce') / 86400.0)
            
            elif 'exposure' in df.columns: 
                # Approximate fallback for NICER/NuSTAR if 'end_time' is missing
                logger.warning(f"'{stop_time_column}' not found. Falling back to APPROXIMATE calculation from 'exposure'.")
                df[stop_time_column] = df['time'] + (pd.to_numeric(df['exposure'], errors='coerce') / 86400.0)

            elif 'exposure_a' in df.columns:
              # Approximate fallback for NICER/NuSTAR if 'end_time' is missing
              logger.warning(f"'{stop_time_column}' not found. Falling back to APPROXIMATE calculation from 'exposure_a'.")
              df[stop_time_column] = df['time'] + (pd.to_numeric(df['exposure_a'], errors='coerce') / 86400.0)
            
            else:
                # If we cannot calculate a stop time at all, exit gracefully.
                logger.error(f"Cannot determine stop time: Missing '{stop_time_column}', 'duration', and 'exposure' columns.")
                return pd.DataFrame()

        # 5. Final data cleaning before use.
        df.dropna(subset=['time', stop_time_column], inplace=True)
        df['obsid'] = df['obsid'].astype(str)
        df["date"] = Time(df["time"], format="mjd").to_datetime()
        df_sorted = df.sort_values(by="time").reset_index(drop=True)

        if df_sorted.empty: return pd.DataFrame()

        # This list comprehension is now protected by the checks above.
        observation_times = [(row['obsid'], row['time'], row[stop_time_column]) for _, row in df_sorted.iterrows()]
        
        # ... (Rest of your outburst grouping logic is unchanged)
        outbursts, outburst_mapping = [], {}
        if observation_times:
            current_outburst = [observation_times[0]]
            for obs in observation_times[1:]:
                last_end = max([o[2] for o in current_outburst])
                if obs[1] < last_end + 182.5:
                    current_outburst.append(obs)
                else:
                    outbursts.append(current_outburst)
                    current_outburst = [obs]
            outbursts.append(current_outburst)
            
        for i, outburst in enumerate(outbursts):
            for obs_id, _, _ in outburst:
                outburst_mapping[obs_id] = i + 1
                
        df_sorted['outburst_id'] = df_sorted['obsid'].map(outburst_mapping)
        output_df = df_sorted[['obsid', 'time', 'date', 'outburst_id', 'ra', 'dec']].rename(
            columns={'obsid': 'OBSID', 'time': 'MJD', 'date': 'DATE', 'outburst_id': 'OUTBURST_ID', 'ra': 'RA', 'dec': 'DEC'})
            
        return output_df
        
def heasarc_pipeline_runner(source: str, obsids: list, base_outdir: str, mission: str, hdf5_file_path: str, status_pane, outburst_mapping: dict, custom_flags: dict = None):
    """
    Orchestrates the full download and analysis pipeline for a list of observations.

    This function iterates through each provided Observation ID (OBSID). For each one,
    it determines the appropriate processing flags, calls a wrapper to download
    and process the raw data from HEASARC, finds the resulting event file, and
    then calls the main analysis function (`process_observation`) to generate
    scientific products (light curves, PDS) and save them to a central HDF5 file.

    Args:
        source (str): The name of the astronomical source (e.g., 'IGR J17091-3624').
        obsids (list): A list of observation IDs to be processed.
        base_outdir (str): The base directory where raw data for each observation
                           will be temporarily downloaded.
        mission (str): The name of the mission (e.g., 'nicer').
        hdf5_file_path (str): The full path to the output HDF5 file where results
                              will be stored.
        status_pane (pn.pane.Markdown): A Panel widget used to display real-time
                                        progress updates to the user.
        outburst_mapping (dict): A dictionary mapping each OBSID to its
                                           pre-calculated outburst number.
        custom_flags (dict): A dictionary of extra processing flags to override the defaults.

    Side effects:
        - Updates the `status_pane` UI element with the current processing status.
        - Downloads a large amount of data from HEASARC into the `base_outdir`.
        - Creates and modifies the specified `hdf5_file_path` with analysis results.
        - Deletes the raw data directories after processing if `TEST_MODE` is `False`.
    """
    logger.info(f"PIPELINE RUNNER STARTED for {len(obsids)} OBSIDs.")
    valid_obsids, zero_expo_obsids = partition_obsids_by_exposure(obsids, mission)

    for i, obsid in enumerate(obsids):
        status_pane.object = f"Processing OBSID {obsid} ({i+1}/{len(obsids)})..."
        try:
            final_flags = {}
            if mission == 'rxte':
                final_flags = {"event_file_patterns": ["pca/FP_gx1_*.gz", "pca/FP_gx2_*.gz"]}
            elif mission == 'nicer' and obsid in zero_expo_obsids:
                final_flags = {
                    "threshfilter": "ALL",
                    "underonlyscr": "5.0:1000.0:0:600",
                    "mpugtiscr": "1.5:1.0:0:20",
                    "lowmemscr": "2.0:200:0:300"
                }
            elif mission == 'nustar':
                final_flags = {}

            if custom_flags:
                final_flags.update(custom_flags)

            obs_dir_path = retrieve_heasarc_data_by_obsid_wrapper(
                source=source,
                obsid=obsid, 
                base_outdir=base_outdir,
                mission=mission,
                flags=final_flags
            )
            if not obs_dir_path:
                continue
            if mission == 'nicer':
             evt_files = glob.glob(os.path.join(obs_dir_path, '**', '*_cl_bary.evt'), recursive=True)
            elif mission == 'nustar':
             evt_files = glob.glob(os.path.join(obs_dir_path,  '**', '*_src1_bary.evt'), recursive=True)
            elif mission == 'rxte':
             evt_files = glob.glob(os.path.join(obs_dir_path,  '**', '*_cl_evt.fits'), recursive=True)
                
            if not evt_files:
                logger.error(f"No 'evt' file found for OBSID {obsid} in {obs_dir_path}. Cannot process.")
                continue
            
            # Get the specific outburst ID for this observation
            outburst_id = outburst_mapping.get(obsid, 0) # Default to 0 if not found

            logger.info(f"Appending data from {obsid} (Outburst {outburst_id}) to {hdf5_file_path}")
            
            # Pass the outburst ID to the processing function
            asyncio.run(process_observation(evt_files, hdf5_file_path, outburst_id, mission))

            if not TEST_MODE:
                shutil.rmtree(obs_dir_path)
            else:
                logger.info(f"TEST_MODE is True. Skipping deletion of raw data directory: {obs_dir_path}")

        except Exception as e:
            logger.error(f"Failed to process observation {obsid}. Error: {e}", exc_info=True)
            continue

    logger.info("PIPELINE COMPLETED.")
    status_pane.object = f"**Pipeline Finished.** HDF5 file updated at: {hdf5_file_path}"
    pn.state.cache['latest_h5_file'] = hdf5_file_path


def update_local_source_list(mission, local_widget, select_all_sources_checkbox):
    """
    Finds local HDF5 files for a given mission and updates the source selection UI.

    This function searches the local 'data/<mission>' directory for processed
    HDF5 files. It then populates a MultiSelect widget with the names of the
    sources found, enabling it and its corresponding "select all" checkbox. If no
    data is found, it updates the widget to reflect that.

    Args:
        mission (str): The name of the mission directory to search (e.g., 'nicer').
        local_widget (pn.widgets.MultiSelect): The widget that displays the list
                                               of available local sources.
        select_all_sources_checkbox (pn.widgets.Checkbox): The checkbox used to
                                                            select all sources in the widget.

    Side effects:
        - Modifies the `.options`, `.disabled`, and `.name` attributes of the `local_widget`.
        - Modifies the `.disabled` and `.value` attributes of the `select_all_sources_checkbox`.
    """
    # --- RESET WIDGETS ---
    print(f"DEBUG: Searching for local data from CWD: {os.getcwd()}")
    local_widget.options = []
    local_widget.disabled = True
    select_all_sources_checkbox.disabled = True
    select_all_sources_checkbox.value = False

    main_data_dir = "data"
    #main_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    source_list = []
    mission_dir = os.path.join(main_data_dir, mission)

    if not os.path.exists(mission_dir):
        local_widget.name = "No local data found"
        return

    h5_files = sorted(glob.glob(os.path.join(mission_dir, '*.h5')))
    for f_path in h5_files:
        source_name = os.path.basename(f_path).replace('.h5', '')
        source_list.append(source_name)

    if source_list:
        local_widget.options = source_list
        local_widget.name = f"Select Processed Source(s) ({len(source_list)} found)"
        local_widget.disabled = False
        # --- ENABLE THE CHECKBOX ---
        select_all_sources_checkbox.disabled = False
    else:
        local_widget.name = "No local data found"


def load_local_source_data(event, outburst_widget, obsid_widget, name_widget, tels_name_widget, run_button_widget, select_all_checkbox):
    """
    Loads HID data from a local HDF5 file to populate and enable UI widgets.

    This function is a callback triggered when a user selects a source from the
    local database. It reads the HID data table from the corresponding HDF5 file,
    cleans it, and then uses the data to populate the outburst and OBSID
    selection widgets. It also enables the relevant UI controls.

    Args:
        event (param.Event): The event object from the source selection widget,
                             containing the list of selected sources.
        outburst_widget (pn.widgets.MultiSelect): The widget for selecting outbursts.
        obsid_widget (pn.widgets.MultiSelect): The widget for selecting OBSIDs.
        name_widget (pn.widgets.TextInput): The main source name input widget, which
                                            is updated with the selected source.
        tels_name_widget (pn.widgets.Select): The widget indicating the current telescope.
        run_button_widget (pn.widgets.Button): The main "Download & Process" button.
        select_all_checkbox (pn.widgets.Checkbox): The checkbox for selecting all outbursts.

    Side effects:
        - Resets and populates the `outburst_widget` and `obsid_widget`.
        - Updates the `name_widget` with the name of the selected source.
        - Enables the `outburst_widget`, `obsid_widget`, `run_button_widget`, and
          `select_all_checkbox`.
        - Caches the loaded and cleaned DataFrame in `pn.state.cache['local_hid_df']`.

    Restrictions:
        - The function will only proceed if exactly one source is selected.
    """
    # --- 1. Clear local widgets when the source changes ---
    outburst_widget.options, outburst_widget.value = [], []
    obsid_widget.options, obsid_widget.value = [], []
    outburst_widget.disabled, obsid_widget.disabled = True, True
    
    # -- ADDED -- Reset checkbox state
    select_all_checkbox.disabled = True
    select_all_checkbox.value = False
    
    pn.state.cache['local_hid_df'] = None
    
    # Only proceed if a single source is selected
    if len(event.new) > 1:
        outburst_widget.name = " Select only ONE source."
        return
    if not event.new:
        outburst_widget.name = "Select Outburst(s)"
        return

    # --- 2. Prepare to read the selected file ---
    outburst_widget.name = "Select Outburst(s)"
    selected_source = event.new[0]
    mission = tels_name_widget.value
    h5_path = os.path.join("data", mission, f"{selected_source}.h5")

    if os.path.exists(h5_path):
        #name_widget.value = selected_source
        
        with h5py.File(h5_path, 'r') as hdf:
            if 'hid/hid_table' not in hdf:
                logger.error(f"'hid/hid_table' not found in {h5_path}")
                outburst_widget.name = "Error: H5 table not found."
                return
            
            # --- 3. Read and CLEAN data ---
            csv_bytes = hdf['hid/hid_table'][0]
            csv_str = csv_bytes.decode('utf-8')
            hid_df = pd.read_csv(pd.io.common.StringIO(csv_str), dtype={'ObsID': str})

            hid_df.dropna(subset=['ObsID', 'Outburst'], inplace=True)
            hid_df['Outburst'] = pd.to_numeric(hid_df['Outburst'], errors='coerce')
            hid_df.dropna(subset=['Outburst'], inplace=True)

            hid_df['Outburst'] = hid_df['Outburst'].astype('int64')

            # --- 4. Populate the UI with the CLEANED data ---
            pn.state.cache['local_hid_df'] = hid_df
            
            outbursts = sorted(hid_df['Outburst'].unique().tolist())
            outburst_widget.options = [f"Outburst {o}" for o in outbursts]
            outburst_widget.disabled = False
            
            # -- ADDED -- Enable the checkbox
            select_all_checkbox.disabled = False
            obsid_options = {
            f"{row['ObsID']} (Outburst {int(row['Outburst'])})": str(row['ObsID'])
            for _, row in hid_df.iterrows()
            }
            obsid_widget.options = obsid_options
            obsid_widget.value = [] 
            obsid_widget.disabled = False
            
            run_button_widget.disabled = False

def filter_obsids_by_outburst(event, obsid_widget, cache_key, outburst_col_name, obsid_col_name):
    """
    Filters the contents of an OBSID widget based on a user's outburst selection.

    This function acts as a generic callback. When a user selects one or more
    outbursts from a widget, this function retrieves the relevant DataFrame from
    the application cache, filters it to find all observations belonging to those
    outbursts, and updates the OBSID selection widget to show only those results.

    Args:
        event (param.Event): The event object from the outburst selection widget,
                             containing the list of selected outburst strings.
        obsid_widget (pn.widgets.MultiSelect): The OBSID widget whose selection
                                               will be updated by this function.
        cache_key (str): The key used to retrieve the source DataFrame from the
                         `pn.state.cache`.
        outburst_col_name (str): The name of the column in the cached DataFrame
                                 that contains the outburst numbers.
        obsid_col_name (str): The name of the column in the cached DataFrame that
                              contains the observation IDs.

    Side effects:
        - Modifies the `.value` of the `obsid_widget` to reflect the filtered list
          of observation IDs.
    """
    df = pn.state.cache.get(cache_key)
    if df is None or df.empty:
        return

    selected_outburst_numbers = [int(item.split(" ")[1]) for item in event.new]

    if not selected_outburst_numbers:
        obsid_widget.value = []
        return

    df_copy = df.copy()
    df_copy[outburst_col_name] = pd.to_numeric(df_copy[outburst_col_name], errors='coerce')
    
    selected_rows = df_copy[df_copy[outburst_col_name].isin(selected_outburst_numbers)]

    obsid_widget.value = selected_rows[obsid_col_name].astype(str).tolist()

    
#def create_pipeline_runner_tab(status_pane):
def create_pipeline_runner_tab(status_pane, plot_local_hids_callback, plot_global_hid_callback):
    """
    Creates the main UI tab for the data processing and visualization pipeline.

    This function initializes all the necessary Panel widgets (e.g., buttons,
    selectors, text inputs), arranges them into a logical layout with a HEASARC
    query section and a local database section, and connects each interactive
    widget to its corresponding callback function.

    Args:
        status_pane (pn.pane.Markdown): The pane used for displaying status updates
                                        to the user.

    Returns:
        tuple: A tuple containing the two main components of the UI:
        - The first element is a Panel layout object (`pn.Column`) containing all
          the control widgets.
        - The second element is a Panel layout object (`pn.Column`) that serves as
          the main display area for plots and results.
    """

        # --- Define CSS Styles ---
    input_style = """
    /* This styles the input box itself */
    :host .bk-input {
        background-color: #f2f2f2 !important;
        border: 1px solid #000000 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        padding-left: 10px !important;
        box-shadow: inset 0px 4px 8px rgba(0, 0, 0, 0.1) !important; /* <-- ADDED SHADOW */
    }

    /* This new rule styles the label (the 'name' text) */
    :host .bk-input-group label {
        font-size: 14px !important;
        color: #000000 !important;
    }
    """

    secondary_button_style = """
    :host .bk-btn {
        background-color: #ffffff !important;
        color: black !important;
        border: 1px solid #000000 !important;
        border-radius: 6px !important;  
        height: 40px !important; 

    }

    :host .bk-btn:hover {
        background-color: #e9e9e9 !important;
    }
    """

    # --- Widget Definitions ---
    tels_name = pn.widgets.Select(
        name="Select Telescope",
        options=['nicer', 'nustar', 'rxte'], 
        sizing_mode='stretch_width', 
        height=60 , 
        stylesheets=[input_style]
    )  
    
    heasarc_heading = pn.pane.Markdown("### Data available on HEASARC")
    source_name = pn.widgets.TextInput(
        name="Enter Source Name", 
        placeholder='e.g., IGR J17091-3624',
        sizing_mode='stretch_width',
        height=60 ,
        stylesheets=[input_style]
    )
    
    fetch_obsids_button = pn.widgets.Button(
        name="Fetch Available OBSID's", 
        button_type="default", 
        sizing_mode='stretch_width',
        stylesheets=[secondary_button_style]
    )
    
    outburst_summary_pane = pn.pane.Markdown("")
    outburst_select = pn.widgets.MultiSelect(
        name="Select Whole Outburst(s)",
        disabled=True, 
        sizing_mode='stretch_width', 
        height=70, 
        stylesheets=[input_style]
    )
    
    multi_obsid_select = pn.widgets.MultiSelect( 
        disabled=True,
        sizing_mode='stretch_width', 
        height=180,
        stylesheets=[input_style]
    )

    local_db_heading = pn.pane.Markdown("### Local Database")
    local_source_select = pn.widgets.MultiSelect(
        name="Select Processed Source(s)", 
        disabled=True, 
        sizing_mode='stretch_width', 
        height=150, 
        stylesheets=[input_style]
    )
    
    select_all_sources_checkbox = pn.widgets.Checkbox(
        name="Select All Sources",
        value=False,
        disabled=True
    )
    
    plot_hid_button = pn.widgets.Button(
        name="Plot HID",
        button_type="default",
        disabled=True, 
        sizing_mode='stretch_width',
        stylesheets=[secondary_button_style]
    )
    
    plot_global_hid_button = pn.widgets.Button(
        name="Plot Global HID",
        button_type="default",
        disabled=True,
        sizing_mode='stretch_width',
        stylesheets=[secondary_button_style]
    )

    local_outburst_select = pn.widgets.MultiSelect(
        name="Select Outburst(s)", 
        disabled=True,
        sizing_mode='stretch_width', 
        height=60, 
        stylesheets=[input_style]
    )
    
    select_all_outbursts_checkbox = pn.widgets.Checkbox(
        name="Select All Outbursts", 
        value=False,
        disabled=True
    )
    
    local_obsid_select = pn.widgets.MultiSelect(
        name="OBSIDs in Selected File", 
        size=6, 
        disabled=True,
        sizing_mode='stretch_width',
        height=180, 
        stylesheets=[input_style]
    )
    
    custom_flags_input = pn.widgets.TextAreaInput(
        name='Pass additional flags as a JSON dictionary:', 
        placeholder='{\n  "threshfilter": "ALL"\n}', 
        resizable="height",
        sizing_mode='stretch_width', 
        stylesheets=[input_style]
    )
    
    run_button = pn.widgets.Button(
        name="Download & Process Observations",
        button_type="success",
        disabled=True,
        sizing_mode='stretch_width',
        stylesheets=[secondary_button_style]
    )

    # --- Callbacks ---
    async def fetch_obsids_callback(event):
        """Callback to fetch and display observation data from HEASARC."""
        fetch_obsids_button.loading = True
        status_pane.object = "Fetching and grouping OBSIDs..."
        try:
            outburst_df = await asyncio.to_thread(get_obsids_from_heasarc_direct, source_name.value, tels_name.value)
            pn.state.cache['outburst_df'] = outburst_df
            if outburst_df.empty:
                multi_obsid_select.name, multi_obsid_select.options = "No observations found.", []
                run_button.disabled, outburst_select.disabled, outburst_summary_pane.object = True, True, ""
                return
            
            options_dict = {f"{row['OBSID']} (Outburst {row['OUTBURST_ID']})": str(row['OBSID']) for _, row in outburst_df.iterrows()}
            multi_obsid_select.options = options_dict
            
            multi_obsid_select.name = f"Available Observations ({len(outburst_df)} found)"
            multi_obsid_select.disabled, run_button.disabled = False, False
            num_outbursts = outburst_df['OUTBURST_ID'].max()
            outburst_summary_pane.object = f"**Source has {int(num_outbursts)} detected outburst(s).**"
            outburst_options = [f"Outburst {i}" for i in range(1, int(num_outbursts) + 1)]
            outburst_select.options, outburst_select.disabled = outburst_options, False
            status_pane.object = "Status: Ready to run pipeline."
            
        except NameResolveError:
            error_message = (
            f"The name '{source_name.value}' is incorrect or could not be found for {tels_name.value}.<br>"
            "Please enter the correct source name."
        )
            status_pane.object = error_message
            multi_obsid_select.name, multi_obsid_select.options = "Enter a valid source name.", []
            run_button.disabled, outburst_select.disabled, outburst_summary_pane.object = True, True, ""
            
        finally:
            fetch_obsids_button.loading = False

    async def run_pipeline_callback(event):
        run_button.disabled = True
        status_pane.object = "Status: Pipeline is starting..."
        try:
            heasarc_obsids = set(multi_obsid_select.value)
            #local_obsids = set(local_obsid_select.value)
            selected_obsids = list(heasarc_obsids)
            
            raw_source_name = source_name.value
            base_outdir = raw_source_name.replace(" ", "_")
            output_dir = os.path.join("data", tels_name.value)
            os.makedirs(output_dir, exist_ok=True)
            hdf5_file_path = os.path.join(output_dir, f"{base_outdir}.h5")
            hdf5_file_path = os.path.abspath(hdf5_file_path)
            os.makedirs(os.path.dirname(hdf5_file_path), exist_ok=True)


            if not selected_obsids:
                status_pane.object, run_button.disabled = "Status: No OBSIDs selected.", False
                return

            # Get the outburst mapping from the cached DataFrame
            outburst_df = pn.state.cache.get('outburst_df')
            if outburst_df is None or outburst_df.empty:
                status_pane.object = "Error: Outburst data not found. Please fetch OBSIDs first."
                run_button.disabled = False
                return
            
            # Create the mapping dictionary
            outburst_mapping = pd.Series(outburst_df.OUTBURST_ID.values, index=outburst_df.OBSID.astype(str)).to_dict()

            custom_flags = {}
            custom_flags_str = custom_flags_input.value.strip()
            if custom_flags_str:
                try:
                    custom_flags = json.loads(custom_flags_str)
                    if not isinstance(custom_flags, dict): raise ValueError("Input must be a JSON dictionary.")
                except (json.JSONDecodeError, ValueError) as e:
                    status_pane.object, run_button.disabled = f"Error: Invalid JSON for custom flags. {e}", False
                    return

            # Pass the new outburst_mapping to the runner
            await asyncio.to_thread(
                heasarc_pipeline_runner, source=raw_source_name, obsids=selected_obsids,
                base_outdir=base_outdir, mission=tels_name.value, hdf5_file_path=hdf5_file_path,
                status_pane=status_pane, custom_flags=custom_flags,
                outburst_mapping=outburst_mapping
            )
            update_local_source_list(tels_name.value, local_source_select, select_all_sources_checkbox)
        except Exception as e:
            status_pane.object = f"Pipeline Failed: {e}"
        finally:
            run_button.disabled = False

 
    def on_plot_hid_click(event):
        """Callback to generate and display local Hardness-Intensity Diagrams (HIDs)."""
        selected_sources = local_source_select.value
        selected_obsids = local_obsid_select.value
        selected_mission = tels_name.value

        # CASE 1: User has selected specific observations. Plot only those.
        if selected_obsids:            
            # This logic requires that exactly one source file is active.
            logger.info("--> BRANCH: Handling custom ObsID selection.")
            if len(selected_sources) != 1:
                return

            source_name = selected_sources[0]
            full_df = pn.state.cache.get('local_hid_df')
            if full_df is None or full_df.empty:
              return
            logger.info(f"--> Full dataframe from cache has {full_df.shape[0]} rows.")
                
            filtered_df = full_df[full_df['ObsID'].astype(str).isin(selected_obsids)].copy()
            logger.info(f"--> Filtered dataframe now has {filtered_df.shape[0]} rows. This is what should be plotted.")
            logger.info("--> Calling plot_local_hids_callback with the filtered dataframe.")

            plot_local_hids_callback(event, selected_sources, hid_df=filtered_df, mission=selected_mission)

        else:
            logger.info("--> BRANCH: Handling full source plot.")
            if not selected_sources:
                
             return
            logger.info("--> Calling plot_local_hids_callback for full source(s).")
            plot_local_hids_callback(event, selected_sources, selected_mission)
  
    def on_plot_global_hid_click(event):
        """Callback to generate and display a global HID for multiple sources."""
        selected_sources = local_source_select.value
        selected_mission = tels_name.value
        if not selected_sources:
            status_pane.object = "Status: No sources selected to plot a global HID."
            return
        status_pane.object = f"Status: Plotting a global HID for: {', '.join(selected_sources)}"
        plot_global_hid_callback(event, selected_sources, selected_mission)
        

    # --- NEW FUNCTION TO MANAGE BUTTON STATE ---
    def update_plot_button_state(event):
        """Enables or disables plotting buttons based on source selection."""
        # event.new is the list of selected sources
        are_sources_selected = bool(event.new)
        plot_hid_button.disabled = not are_sources_selected
        plot_global_hid_button.disabled = not are_sources_selected

    def toggle_all_sources(event):
        """Callback to select or deselect all items in the source widget."""
        if event.new:
            local_source_select.value = local_source_select.options
        else:
            local_source_select.value = []

    def toggle_all_outbursts(event):
        """Callback to select or deselect all items in the outburst widget."""
        if event.new:
            local_outburst_select.value = local_outburst_select.options
        else:
            local_outburst_select.value = []

    # --- Layout Construction ---
    left_column = pn.Column(
        heasarc_heading,
        source_name,
        fetch_obsids_button,
        outburst_summary_pane, 
        outburst_select,
        multi_obsid_select, custom_flags_input, run_button
    )
    
    # --- 3. ADD BUTTONS TO THE LAYOUT ---
    plot_button_row = pn.Row(plot_hid_button, plot_global_hid_button)
    
    right_column = pn.Column(
        local_db_heading, 
        local_source_select, 
        local_outburst_select, 
        select_all_outbursts_checkbox,
        local_obsid_select,
        select_all_sources_checkbox,
        plot_button_row,  
        
    )
    divided_layout = pn.Row(left_column, right_column)
    controls = pn.Column(
         tels_name, divided_layout
    )
    
    # --- 4. CONNECT EVERYTHING ---
    fetch_obsids_button.on_click(fetch_obsids_callback)
    
    outburst_select.param.watch(
        lambda e: filter_obsids_by_outburst(
            e, multi_obsid_select, 'outburst_df', 'OUTBURST_ID', 'OBSID'
        ), 
        'value'
    )
    
    run_button.on_click(run_pipeline_callback)

    tels_name.param.watch(
        lambda e: update_local_source_list(e.new, local_source_select, select_all_sources_checkbox), 
        'value'
    )
    
    local_source_select.param.watch(
        lambda e: load_local_source_data(
            e, local_outburst_select, local_obsid_select, source_name, 
            tels_name, run_button, select_all_outbursts_checkbox
        ), 
        'value'
    )
    
    local_outburst_select.param.watch(
        lambda e: filter_obsids_by_outburst(
            e, local_obsid_select, 'local_hid_df', 'Outburst', 'ObsID'
        ), 
        'value'
    )
    
    select_all_sources_checkbox.param.watch(toggle_all_sources, 'value')
    select_all_outbursts_checkbox.param.watch(toggle_all_outbursts, 'value')

    # --- CONNECT NEW BUTTONS AND STATE MANAGER ---
    plot_hid_button.on_click(on_plot_hid_click)
    plot_global_hid_button.on_click(on_plot_global_hid_click)
    local_source_select.param.watch(update_plot_button_state, 'value')
    
    update_local_source_list(
        tels_name.value, 
        local_source_select,
        select_all_sources_checkbox
    )
    
    return controls, tels_name
