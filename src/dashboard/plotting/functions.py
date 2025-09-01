import panel as pn
import param
import h5py
import pandas as pd
import io
import os
import plotly.express as px
import numpy as np
from PIL import Image
import logging # IMPORT THE LOGGING MODULE
import glob
from matplotlib.ticker import MaxNLocator
logger = logging.getLogger(__name__) # INITIALIZE LOGGER

E_bands = {
    "nustar": {
        "soft_Eband": (3, 6),
        "high_Eband": (6, 10),
        "full_Eband": (3, 79),
    },
    "nicer": {
        "soft_Eband": (2, 4),
        "high_Eband": (4, 12),
        "full_Eband": (0.4, 12),
    },
    "rxte": {
        "soft_Eband": (3, 6),
        "high_Eband": (6, 15),
        "full_Eband": (3, 20),
    },
}

class HIDPlotter(param.Parameterized):
    h5_file_path = param.String(default="", doc="Path to the HDF5 file.")
    hid_df = param.DataFrame(default=pd.DataFrame(), allow_None=True)

    def __init__(self, hid_df=None, **params):
        super().__init__(**params)
        if hid_df is not None:
            logger.info(f"HIDPlotter __init__: Using provided dataframe with {hid_df.shape[0]} rows.")
            self.hid_df = hid_df
        else:
            logger.info("HIDPlotter __init__: No dataframe provided, loading from file.")
            self.hid_df = self.get_hid_data(self.h5_file_path)
        
    @staticmethod
    def get_hid_data(h5_file_path):
        if not (h5_file_path and os.path.exists(h5_file_path)):
            logger.warning(f"HID data file not found: {h5_file_path}") 
            return None
        try:
            with h5py.File(h5_file_path, 'r') as hdf:
                if 'hid/hid_table' not in hdf:
                    logger.warning(f"No 'hid/hid_table' found in HDF5 file: {h5_file_path}") 
                    return None
                csv_str = hdf['hid/hid_table'][0].decode('utf-8')
                df = pd.read_csv(io.StringIO(csv_str))
                required_cols = ['ObsID', 'Hardness_Ratio', 'Intensity', 'Outburst', 'No_of_detectors', 'Normalized_Intensity']
                if not all(col in df.columns for col in required_cols):
                    # Fallback for older H5 files not yet re-processed
                    if 'Intensity' in df.columns and 'No_of_detectors' in df.columns and 'Normalized_Intensity' not in df.columns:
                        df['Normalized_Intensity'] = df['Intensity'] / df['No_of_detectors']
                    if 'No_of_detectors' not in df.columns:
                        df['No_of_detectors'] = np.nan
                        logger.warning(f"'{h5_file_path}': 'No_of_detectors' column missing in HID data. Filling with NaN.")
                    if 'Normalized_Intensity' not in df.columns:
                        df['Normalized_Intensity'] = np.nan
                        logger.warning(f"'{h5_file_path}': 'Normalized_Intensity' column missing in HID data. Filling with NaN.") 

                    if not all(col in df.columns for col in required_cols): # Re-checking after fallbacks
                        logger.error(f"'{h5_file_path}': Missing critical columns even after fallback. Required: {required_cols}") 
                        return None

                for col in ['Hardness_Ratio', 'Intensity', 'Outburst', 'No_of_detectors', 'Normalized_Intensity']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=['Hardness_Ratio', 'Normalized_Intensity'], inplace=True)
                df['ObsID'] = df['ObsID'].astype(str)
                return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error in get_hid_data for {h5_file_path}: {e}", exc_info=True) 
            return None

    @staticmethod
    def get_all_lightcurve_pngs(h5_file_path, obs_id):
        """
        Scans the HDF5 file and retrieves all light curve plots for a given ObsID.
        Returns a dictionary where keys are clean titles and values are PNG data.
        """
        lc_plots = {}
        try:
            with h5py.File(h5_file_path, 'r') as hdf:
                obs_group = hdf.get(obs_id)
                if not obs_group:
                    logger.warning(f"No group found for ObsID {obs_id} in {h5_file_path}")
                    return {}
                
                # Loops through all datasets in the observation's group
                for item_name in obs_group:
                    # Checks if the dataset name indicates it's a light curve plot
                    if item_name.startswith("lightcurve_"):
                        png_data = obs_group[item_name][()]
                        
                        # Create a clean title for the plot from the dataset name
                        # e.g., "lightcurve_0.4-12_keV_part1" becomes "0.4-12 keV part1"
                        clean_title = item_name.replace("lightcurve_", "").replace("_", " ")
                        lc_plots[clean_title] = png_data
        except Exception as e:
            logger.error(f"Error getting all lightcurves for {obs_id}: {e}", exc_info=True)
        
        return lc_plots

    @staticmethod
    def get_pds_png(h5_file_path, obs_id):
        try:
            with h5py.File(h5_file_path, 'r') as hdf:
                dataset_path = f"{obs_id}/pds"
                if dataset_path in hdf:
                    return hdf[dataset_path][()]
        except Exception as e:
            logger.error(f"Error getting PDS for {obs_id}: {e}", exc_info=True) 
        return None

    @staticmethod
    def get_pds_attributes(h5_file_path, obs_id):
        try:
            with h5py.File(h5_file_path, 'r') as hdf:
                pds_plot_name = "pds_leahy_plot"
                dataset_path = f"{obs_id}/{pds_plot_name}"
                if dataset_path in hdf:
                    return dict(hdf[dataset_path].attrs) # Return dictionary of attributes
        except Exception as e:
            logger.error(f"Error getting PDS attributes for {obs_id}: {e}", exc_info=True) 
        return {} # Return empty dict on error or not found
        
    @param.depends('hid_df')
    def hid_plot(self, mission, xscale='linear', yscale='log', hid_df=None):
        source_df = self.hid_df if hid_df is None else hid_df
        if source_df is None or source_df.empty:
            return pn.pane.Markdown("### No valid data to plot. Please select sources.")

        # Using a copy to avoid modifying the original dataframe
        data_to_plot = source_df.copy()
        
        data_to_plot.sort_values('Outburst', inplace=True)
        data_to_plot['Outburst'] = data_to_plot['Outburst'].astype('category')

        bands = E_bands[mission]
        soft_band = bands["soft_Eband"]
        hard_band = bands["high_Eband"]
        full_band = bands["full_Eband"]

        custom_data_cols = ['ObsID', 'Outburst', 'Hardness_Ratio', 'Intensity', 'No_of_detectors', 'Normalized_Intensity']

        fig = px.scatter(
            data_to_plot,
            x='Hardness_Ratio',
            y='Normalized_Intensity',
            color='Outburst',
            hover_name='ObsID',
            # Pass custom_data directly into the creation call
            custom_data=custom_data_cols,
            labels={
                'Hardness_Ratio': f'Hardness Ratio ({hard_band[0]}-{hard_band[1]} keV / {soft_band[0]}-{soft_band[1]} keV)',
                'Normalized_Intensity': f'Normalized Intensity ({full_band[0]}-{full_band[1]} keV) [cts/s/detector]'
            }
        )
        
        if yscale == 'log':
           fig.update_yaxes(dict(type = yscale, dtick = 1))
        else:
           fig.update_yaxes(type = yscale)
       
        if xscale == 'log':
            fig.update_xaxes(dict(type = xscale, dtick = 1))
        else:
            fig.update_xaxes(type = xscale)

        fig.update_traces(
            marker=dict(size=8, line=dict(width=1, color='white')),
            hovertemplate=(
                "<b>ObsID:</b> %{customdata[0]}<br>"
                "<b>Outburst:</b> %{customdata[1]}<br>"
                "<b>HR:</b> %{customdata[2]:.3f}<br>"
                "<b>Intensity:</b> %{customdata[3]:.2f} cts/s<br>"
                "<b>Detectors:</b> %{customdata[4]}<br>"
                "<b>Norm. Int.:</b> %{customdata[5]:.3f} cts/s/det<extra></extra>"
            )
        )

        fig.update_layout(
            width=450, height=400, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(color='black'), hovermode='closest', margin=dict(l=40, r=20, t=20, b=40)
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)

        plotly_pane = pn.pane.Plotly(fig, name=self.name, config={'displaylogo': False})

        return plotly_pane

def get_global_hid_data(selected_sources, main_data_dir, mission):
    """
    Aggregates HID data from multiple HDF5 files into a single DataFrame.
    """
    all_dfs = []
    for source_name in selected_sources:
        search_pattern = os.path.join(main_data_dir, mission, '**', f"{source_name.replace(' ', '_')}.h5")
        found_files = glob.glob(search_pattern, recursive=True)
        # 1. Checks if the LIST is empty, not if the list "exists".
        # 2. If files were found, take the FIRST element from the list.
        if not found_files:
            print(f"Warning: HDF5 file not found for {source_name}. Skipping.")
            continue
        
        # Get the first file path from the list
        h5_path = found_files[0] 
        
        # Use the existing static method from HIDPlotter to get the data
        df = HIDPlotter.get_hid_data(h5_path)
        
        if df is not None and not df.empty:
            df['source_name'] = source_name  # Add the source name column for coloring
            all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()  # Return empty if no data was found
        
    # Combine all dataframes into one global dataframe
    global_df = pd.concat(all_dfs, ignore_index=True)
    return global_df

def create_global_hid_plot(global_df, mission, xscale='linear', yscale='log'):
    """
    Generates an interactive global HID plot.
    """
    if global_df.empty:
        return pn.pane.Markdown("### No valid data to plot. Please select sources.")

    global_df['Outburst'] = global_df['Outburst'].astype('category')
    
    bands = E_bands[mission]
    soft_band = bands["soft_Eband"]
    hard_band = bands["high_Eband"]
    full_band = bands["full_Eband"]

    fig = px.scatter(
        global_df,
        x='Hardness_Ratio',
        y='Normalized_Intensity',
        color='source_name',
        symbol='Outburst',
        hover_name='ObsID',
        custom_data=['ObsID', 'source_name', 'Outburst', 'Intensity', 'Normalized_Intensity'],
    )
    if yscale == 'log':
        fig.update_yaxes(type=yscale, dtick=1)
    else:
        fig.update_yaxes(type=yscale)

    if xscale == 'log':
        fig.update_xaxes(type=xscale, dtick=1)
    else:
        fig.update_xaxes(type=xscale)

    # Manually setting the legend group and name for each trace ---
    # grouping all outbursts under a single source name in the legend.
    for trace in fig.data:
        # Get the source name from the custom data associated with the trace
        source_name = trace.customdata[0][1]
        trace.name = source_name
        trace.legendgroup = source_name

    # --- Cleans up the legend to show each source name only once ---_-
    legend_names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in legend_names)
            else legend_names.add(trace.name)
    )

    # Updates hover template and set a uniform marker size
    fig.update_traces(
        hovertemplate=(
            "<b>Source:</b> %{customdata[1]}<br>"
            "<b>ObsID:</b> %{customdata[0]}<br>"
            "<b>Outburst:</b> %{customdata[2]}<br>"
            "<b>HR:</b> %{x:.3f}<br>"
            "<b>Intensity:</b> %{customdata[3]:.2f} cts/s<br>"
            "<b>Norm. Int.:</b> %{customdata[4]:.3f} cts/s/det<extra></extra>"
        ),
        marker=dict(
            size=8,
            line=dict(width=1, color='White')
        )
    )

    # Final layout adjustments
    fig.update_layout(
        height=650,
        legend_title_text='Sources',
        xaxis_title=f'Hardness Ratio ({hard_band[0]}-{hard_band[1]} keV / {soft_band[0]}-{soft_band[1]} keV)',
        yaxis_title=f'Normalized Intensity ({full_band[0]}-{full_band[1]} keV) [cts/s/detector]',
        margin=dict(b=0)
    )

    plotly_pane = pn.pane.Plotly(fig, config={'displaylogo': False}, sizing_mode='stretch_width')
    
    return plotly_pane