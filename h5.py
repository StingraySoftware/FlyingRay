
import panel as pn
import asyncio
import time
import logging
import os
import glob
import re
import sys
import numpy as np
import h5py
import matplotlib
import base64
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from io import BytesIO, StringIO
import pandas as pd
from stingray import EventList, Lightcurve, AveragedPowerspectrum
from stingray.gti import get_gti_lengths
from datetime import datetime

from plotting import HIDPlotter, get_global_hid_data, create_global_hid_plot
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def save_plot_data_to_hdf5(group, plot_name, plot_data_bytes):
    if plot_data_bytes:
        plot_data_np = np.frombuffer(plot_data_bytes, dtype=np.uint8)
        group.create_dataset(plot_name, data=plot_data_np)

        
def save_dataframe_to_hdf5(group, df, dataset_name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    csv_buffer.close()
    dt = h5py.special_dtype(vlen=str)
    dataset = group.create_dataset(dataset_name, (1,), dtype=dt)
    dataset[0] = csv_str

"""
def create_energy_band_lightcurves(events, obs_id, dt=10):

    plots_data = {}
    energy_bands = {'0.4-12_keV': (0.4, 12)}
    for band_name, (e_min, e_max) in energy_bands.items():
        try:
            energy_mask = (events.energy >= e_min) & (events.energy <= e_max)
            filtered_times = events.time[energy_mask]
            lc = Lightcurve.make_lightcurve(filtered_times, dt=dt, gti=events.gti, mjdref=events.mjdref)
            lc.apply_gtis()

            gti_gap_threshold = 3600  
            
            break_indices = []
            if len(lc.time) > 1:
                time_diffs = np.diff(lc.time)
                break_indices = np.where(time_diffs > gti_gap_threshold)[0]

            if not break_indices.any():
                fig, ax = plt.subplots()
                
                # Plot the single continuous light curve
                if len(lc.time) > 0:
                    ax.plot(lc.time, lc.countrate, color='k', marker='o', markersize=2, linestyle='-', linewidth=0.5)
                
                # Shade small gaps in red
                if lc.gti is not None and len(lc.gti) > 1:
                    for i in range(len(lc.gti) - 1):
                        gap_start = lc.gti[i, 1]
                        gap_end = lc.gti[i + 1, 0]
                        gap_duration = gap_end - gap_start
                        if 0 < gap_duration < gti_gap_threshold:
                            ax.axvspan(gap_start, gap_end, alpha=0.3, color='red', zorder=0)
                

                ax.set_title(f"Light Curve {band_name}: {obs_id} (dt={dt}s)")
                ax.set_ylabel(r'Counts s$^{-1}$')
                ax.set_xlabel(r'Time [s]')


            else:
                segments = []
                start_idx = 0
                for end_idx in break_indices:
                    segments.append((start_idx, end_idx + 1))
                    start_idx = end_idx + 1
                segments.append((start_idx, len(lc.time)))
                
                # Create a subplot for each segment
                fig, axes = plt.subplots(1, len(segments), sharey=True)
                fig.subplots_adjust(wspace=0.05)

                for i, ax in enumerate(axes):
                    start, end = segments[i]
                    ax.plot(lc.time[start:end], lc.countrate[start:end], color='k', marker='o', markersize=2, linestyle='-', linewidth=0.5)

                axes[0].set_ylabel(r'Counts s$^{-1}$')
                for i, ax in enumerate(axes):
                    ax.spines['left'].set_linewidth(3)
                    ax.spines['right'].set_linewidth(3)
                    ax.spines['top'].set_linewidth(3)
                    ax.spines['bottom'].set_linewidth(3)
                    ax.tick_params(which='major', width=2, length=8, pad=10, direction='in', top=True, right=True)
                    ax.tick_params(which='minor', width=2, length=5, direction='in', top=True, right=True)
 
                    d = .015
                    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=2)
                    if i > 0:
                        ax.plot((-d, +d), (-d, +d), **kwargs)
                        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                    if i < len(axes) - 1:
                        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                
                fig.suptitle(f"Light Curve {band_name}: {obs_id} (dt={dt}s)")
                fig.supxlabel('Time [s]')

            # Save the resulting figure to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plots_data[f"lightcurve_{band_name}"] = buf.getvalue()
            plt.close(fig)
            buf.close()
        except Exception as e:
            logger.error(f"Failed to create lightcurve for {obs_id} band {band_name}", exc_info=True)
            
    return plots_data
"""
            
"""
def create_energy_band_lightcurves(events, obs_id, dt=10):
    plots_data = {}
    energy_bands = {'0.4-12_keV': (0.4, 12)}
    for band_name, (e_min, e_max) in energy_bands.items():
        try:
            energy_mask = (events.energy >= e_min) & (events.energy <= e_max)
            filtered_times = events.time[energy_mask]
            lc = Lightcurve.make_lightcurve(filtered_times, dt=dt, gti=events.gti, mjdref=events.mjdref)
            lc.apply_gtis()
            fig, ax = plt.subplots()
           # ax.errorbar(lc.time, lc.countrate, yerr=lc.countrate_err, color='k', fmt='o', markersize=2) 
            #ax.plot#___________________________________________________________________

            gti_gap_threshold = 3600
            
            if len(lc.time) > 1:
                #the time difference between each consecutive point
                time_diffs = np.diff(lc.time)
                # Find where the gap is larger than 1 hour (3600 seconds)
                break_indices = np.where(time_diffs > 3600)[0]

                # Plot each continuous segment
                start_idx = 0
                for end_idx in break_indices:
                    # Plot the segment before the break
                    ax.plot(lc.time[start_idx:end_idx+1], lc.countrate[start_idx:end_idx+1], color='k', marker='o', markersize=2, linestyle='-', linewidth=0.1)
                    # The next segment starts after the break
                    start_idx = end_idx + 1
                
                # Plot the final remaining segment
                ax.plot(lc.time[start_idx:], lc.countrate[start_idx:], color='k', marker='o', markersize=2, linestyle='-', linewidth=0.5)
            elif len(lc.time) == 1:
                # If there's only one point, plot it without a line
                ax.plot(lc.time, lc.countrate, color='k', marker='o', markersize=2)

            
            ax.set_title(f"Light Curve {band_name}: {obs_id} (dt={dt}s)")
            ax.set_ylabel(r'Counts s$^{-1}$')
            ax.set_xlabel(r'Time [s]')
            ax.tick_params(which='major', width=2, length=8, pad=10, direction='in', top=True, right=True)
            ax.tick_params(which='minor', width=2, length=5, direction='in', top=True, right=True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(3)
   
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plots_data[f"lightcurve_{band_name}"] = buf.getvalue()
            plt.close(fig)
            buf.close()
        except Exception as e:
            logger.error(f"Failed to create lightcurve for {obs_id} band {band_name}", exc_info=True)
    return plots_data

"""

def create_energy_band_lightcurves(events, obs_id, dt=10):
    plots_data = {}
    energy_bands = {'0.4-12_keV': (0.4, 12)}
    for band_name, (e_min, e_max) in energy_bands.items():
        try:
            energy_mask = (events.energy >= e_min) & (events.energy <= e_max)
            filtered_times = events.time[energy_mask]
            lc = Lightcurve.make_lightcurve(filtered_times, dt=dt, gti=events.gti, mjdref=events.mjdref)
            lc.apply_gtis()
            
            fig, ax = plt.subplots()

            gti_gap_threshold = 3600  

            if lc.gti is None or len(lc.gti) < 2:
                # If there's only one continuous block of data, plot it all at once.
                ax.plot(lc.time, lc.countrate, color='k', marker='o', markersize=2, linestyle='-', linewidth=0.1)
            else:
                # Plot in segments, breaking the line for gaps > 1 hour.
                start_of_segment = lc.gti[0, 0]
                for i in range(len(lc.gti) - 1):
                    # Check the duration of the gap to the next GTI
                    gap = lc.gti[i + 1, 0] - lc.gti[i, 1]
                    if gap > gti_gap_threshold:
                        # If gap is too long, plot the segment collected so far
                        end_of_segment = lc.gti[i, 1]
                        mask = (lc.time >= start_of_segment) & (lc.time <= end_of_segment)
                        if np.any(mask):
                            ax.plot(lc.time[mask], lc.countrate[mask], color='k', marker='o', markersize=2, linestyle='-', linewidth=0.1)
                        # Start a new segment
                        start_of_segment = lc.gti[i + 1, 0]
                
                # Plot the final segment after the loop
                end_of_final_segment = lc.gti[-1, 1]
                mask = (lc.time >= start_of_segment) & (lc.time <= end_of_final_segment)
                if np.any(mask):
                    ax.plot(lc.time[mask], lc.countrate[mask], color='k', marker='o', markersize=2, linestyle='-', linewidth=0.1)

            ax.set_title(f"Light Curve {band_name}: {obs_id} (dt={dt}s)")
            ax.set_ylabel(r'Counts s$^{-1}$')
            ax.set_xlabel(r'Time [s]')
            ax.tick_params(which='major', width=2, length=8, pad=10, direction='in', top=True, right=True)
            ax.tick_params(which='minor', width=2, length=5, direction='in', top=True, right=True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(3)

            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plots_data[f"lightcurve_{band_name}"] = buf.getvalue()
            plt.close(fig)
            buf.close()

        except Exception as e:
            logger.error(f"Failed to create lightcurve for {obs_id} band {band_name}", exc_info=True)
            
    return plots_data


def create_pds(events, obs_id, segment_size=50.0, dt=0.001):
    
    try:
        total_exposure = np.sum(get_gti_lengths(events.gti))
        if total_exposure < segment_size:
            return None
        if segment_size <= 0:
            return None

        pds = AveragedPowerspectrum.from_events(
            events, segment_size=segment_size, dt=dt, norm="frac", use_common_mean=True
        )
        pds_reb = pds.rebin_log(0.03)

        noise_powers = pds.power[pds.freq > 100]
        P_noise_Nicer = np.mean(noise_powers) if len(noise_powers) > 0 else 0
        y_vals_reb = (pds_reb.power - P_noise_Nicer) * pds_reb.freq
        y_vals_full = (pds.power - P_noise_Nicer) * pds.freq

        fig = plt.figure()
        ax = plt.gca()

        ax.plot(pds.freq, y_vals_full, drawstyle="steps-mid", color="grey", alpha=0.5)
        ax.plot(pds_reb.freq, y_vals_reb, drawstyle="steps-mid", color="k")

        ax.loglog()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"Power $\times$ Frequency [$(\mathrm{rms/mean})^2$]")
        ax.set_title(f"PDS {obs_id}")

        x_lim_bottom = 1. / segment_size
        x_lim_top = 1. / (2. * dt)
        ax.set_xlim(left=x_lim_bottom, right=x_lim_top)

        M = len(noise_powers)

        if M > 0 and np.any(y_vals_reb) and len(pds.freq[pds.freq > 0]) > 0:
            lowest_freq = pds.freq[pds.freq > 0][0]

            y_lower = ( P_noise_Nicer / np.sqrt(M)) * lowest_freq

            y_upper = np.max(y_vals_reb)
            
            if y_upper > y_lower:
                ax.set_ylim(bottom=y_lower, top=y_upper + (0.2 * y_upper))

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_data = buf.getvalue()
        plt.close(fig)

        return plot_data

    except Exception as e:
        logger.error(f"PDS FAILED for {obs_id}", exc_info=True)
        return None
        
def calculate_hid_parameters(events, obsid, no_of_detectors, outburst_id):
    try:
        total_exposure = np.sum(get_gti_lengths(events.gti))
        if total_exposure <= 0: return None
        intensity = len(events.filter_energy_range((0.4, 12)).time) / total_exposure
        soft_rate = len(events.filter_energy_range((2, 4)).time) / total_exposure
        hard_rate = len(events.filter_energy_range((4, 12)).time) / total_exposure
        hardness_ratio = hard_rate / soft_rate if soft_rate > 0 else np.nan
        normalized_intensity = intensity / no_of_detectors if no_of_detectors > 0 else np.nan
        return {'ObsID': obsid,
                'Intensity': intensity,
                'Hardness_Ratio': hardness_ratio,
                'No_of_detectors': no_of_detectors,
                'Normalized_Intensity': normalized_intensity,
                'Outburst': outburst_id 
               }
    except Exception as e:
        logger.error(f"Failed to calculate HID for {obsid}", exc_info=True)
        return None

        
def calculate_count_rate(events):
    try:
        total_exposure = np.sum(get_gti_lengths(events.gti))
        return len(events.time) / total_exposure if total_exposure > 0 else 0.0
    except Exception:
        return 0.0

async def process_observation(evt_filepath, hdf5_file_path, outburst_id):
    """
    Processes a single observation, including its outburst ID, and saves the data.
    If data for the observation already exists, it will be DELETED and REGENERATED.
    """
    match = re.search(r'(\d{10,})', evt_filepath)
    if not match:
        logger.error(f"Could not extract OBSID from event file: {evt_filepath}")
        return

    obs_id = match.group(1)
    logger.info("-" * 60)
    logger.info(f" PROCESSING OBSID: {obs_id} for Outburst: {outburst_id}")

    try:
        events_data = EventList.read(evt_filepath, "hea", additional_columns=["DET_ID"])
        ndet = len(set(events_data.det_id))
        if ndet == 0 or calculate_count_rate(events_data) < 0.1:
            logger.warning(f"Discarding {obs_id} due to low counts or no detectors.")
            return

        lc_task = asyncio.to_thread(create_energy_band_lightcurves, events_data, obs_id)
        pds_task = asyncio.to_thread(create_pds, events_data, obs_id)
        # Pass the outburst_id to the task
        hid_task = asyncio.to_thread(calculate_hid_parameters, events_data, obs_id, ndet, outburst_id)
        
        lc_plots, pds_plot, hid_params = await asyncio.gather(lc_task, pds_task, hid_task)
        

        with h5py.File(hdf5_file_path, 'a') as hdf:
            if obs_id in hdf:
                logger.warning(f"OBSID {obs_id} already exists. Deleting old group to regenerate.")
                del hdf[obs_id]
            
            obs_group = hdf.create_group(obs_id)

            if lc_plots:
                for name, data in lc_plots.items():
                    save_plot_data_to_hdf5(obs_group, name, data)
            if pds_plot:
                save_plot_data_to_hdf5(obs_group, "pds", pds_plot)

            if hid_params:
                hid_group = hdf.require_group('hid')
                
                existing_df = pd.DataFrame()
                if 'hid_table' in hid_group:
                    try:
                        csv_data = hdf['hid/hid_table'][()][0]
                        if isinstance(csv_data, bytes):
                            csv_str = csv_data.decode('utf-8')
                        else:
                            csv_str = csv_data
                        
                        if csv_str and csv_str.strip():
                            existing_df = pd.read_csv(StringIO(csv_str))
                    except Exception as e:
                        logger.warning(f"Could not parse hid_table: {e}. It will be recreated.")
                        existing_df = pd.DataFrame()

                if not existing_df.empty:
                    existing_df['ObsID'] = existing_df['ObsID'].astype(str)
                    existing_df = existing_df[existing_df['ObsID'] != obs_id]
                
                new_row_df = pd.DataFrame([hid_params])
                combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                
                if 'hid_table' in hid_group:
                    del hid_group['hid_table']
                
                save_dataframe_to_hdf5(hid_group, combined_df, 'hid_table')

            logger.info(f" FINISHED and saved/updated data for OBSID: {obs_id}")

    except Exception as e:
        logger.error(f"Failed to process and save OBSID {obs_id}", exc_info=True)

def create_h5_generator_tab(telescope_selector_widget):
    plotter_map = {}
    plot_scale_state = {}
   # main_data_dir = "data"
    main_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    plot_type_heading = pn.pane.Markdown("## *No Plot Selected*", margin=(15, 0, 0, 20))
    header_card = pn.Card(
        plot_type_heading,
        css_classes=['hid-card'],
        collapsible=True,
        sizing_mode='stretch_width'
    )

    plots_display_area = pn.FlexBox(sizing_mode='stretch_width', min_height=480, justify_content='center', align_items='start')
    float_panel_placeholder = pn.Column(sizing_mode='stretch_width') 
   
    details_html_pane = pn.pane.HTML(sizing_mode='stretch_width')
    close_button = pn.widgets.Button(name='Close Details', button_type='primary')
    details_area = pn.Column(details_html_pane, pn.Row(close_button, align='center'), visible=False, sizing_mode='stretch_width')
    close_button.on_click(lambda e: setattr(details_area, 'visible', False))

    def create_details_html(obs_id, selected_row, plotter):
        """
        Generates HTML to display observation details, including light curves and the PDS plot.
        All plots are rendered in a single, flexible row.
        """
        plot_html_parts = []
    
        # 1. Generate Light Curve HTML
        bands = [
            #'0.4-2_keV',
            #'2-3_keV',
            '0.4-12_keV'
        ]
        style = "width: 400px; margin: 10px; text-align: center;"
        for band in bands:
            png_data = plotter.get_lightcurve_png(plotter.h5_file_path, obs_id, band)
            
            if png_data is not None:
                encoded_image = base64.b64encode(png_data).decode('utf-8')
                img_src = f'data:image/png;base64,{encoded_image}'
                plot_html_parts.append(f'<div style="{style}"><h4>Light Curve ({band})</h4><img src="{img_src}" style="width: 100%; height: auto; border: 1px solid #ddd;"></div>')
            else:
                plot_html_parts.append(f'<div style="{style}"><h4>Light Curve ({band})</h4><p>Not available</p></div>')
        
        pds_png_data = plotter.get_pds_png(plotter.h5_file_path, obs_id)
        pds_html_part = ""
        if pds_png_data is not None:
            encoded_pds_image = base64.b64encode(pds_png_data).decode('utf-8')
            pds_img_src = f'data:image/png;base64,{encoded_pds_image}'
            plot_html_parts.append(f'<div style="{style}"><h4>Power Density Spectrum</h4><img src="{pds_img_src}" style="width: 100%; height: auto; border: 1px solid #ddd;"></div>')
        else:
            plot_html_parts.append(f'<div style="{style}"><h4>Power Density Spectrum</h4><p>Not available</p></div>')

        # Combine all plot parts into a single string
        all_plots_html = "".join(plot_html_parts)

        intensity_val = selected_row.get('Intensity', np.nan)
        normalized_intensity_val = selected_row.get('Normalized_Intensity', np.nan)
        detectors_val = selected_row.get('No_of_detectors', np.nan)
        outburst_val = selected_row.get('Outburst', np.nan)

        final_html = f"""
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin: 20px auto; max-width: 1300px;">
            <div>
                <h3>Observation: {obs_id}</h3>
                <p>
                    <strong>Hardness Ratio:</strong> {selected_row['Hardness_Ratio']:.3f}<br>
                    <strong>Intensity:</strong> {intensity_val:.3f} cts/s<br>
                    <strong>Normalized Intensity:</strong> {normalized_intensity_val:.3f} cts/s/detector<br>
                    <strong>No. of Detectors:</strong> {int(detectors_val) if not np.isnan(detectors_val) else 'N/A'}<br>
                    <strong>Outburst:</strong> {int(outburst_val) if not np.isnan(outburst_val) else 'N/A'}
                </p>
            </div>
            <div>
                <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                    {all_plots_html}
                </div>
            </div>
        </div>"""
        return final_html

    def update_float_panel_details(event, plotter):
        try:
            obs_id = str(event.new['points'][0]['customdata'][0])
            lc_png_data = plotter.get_lightcurve_png(plotter.h5_file_path, obs_id, '0.4-12_keV')
            pds_png_data = plotter.get_pds_png(plotter.h5_file_path, obs_id)
            lc_bytes = lc_png_data.tobytes() if lc_png_data is not None else None
            pds_bytes = pds_png_data.tobytes() if pds_png_data is not None else None
            lc_pane = pn.pane.PNG(lc_bytes, width=400) if lc_bytes is not None else pn.pane.Markdown("### Lightcurve not available.")
            pds_pane = pn.pane.PNG(pds_bytes, width=400) if pds_bytes is not None else pn.pane.Markdown("### PDS not available.")
            title_pane = pn.pane.Markdown(f"### Details for ObsID: {obs_id}", align='center')
            content_layout = pn.Column(title_pane, pn.Row(lc_pane, pds_pane, align='center'))
            
            new_float_panel = pn.layout.FloatPanel(
                content_layout, name=f"Details: {obs_id}", contained=False,
                position='center-top', status='normalized', width=850, height=400, 
                config={'headerControls': { 'maximize': 'remove'}}
            )

            float_panel_placeholder.append(new_float_panel)
        except Exception as e:
            logger.error(f"Failed to show FloatPanel details: {e}", exc_info=True)

    def update_static_details_area(event, plotter):
        if not event or not event.new: return
        details_html_pane.object = '<div style="text-align:center; padding: 20px;"><h3>Loading...</h3></div>'
        details_area.visible = True
        async def load_details_html():
            try:
                obs_id = str(event.new['points'][0]['customdata'][0])
                plotter.hid_df['ObsID'] = plotter.hid_df['ObsID'].astype(str)
                point_data_df = plotter.hid_df[plotter.hid_df['ObsID'] == obs_id]
                if point_data_df.empty: raise ValueError(f"Could not find data for {obs_id}")
                details_html_pane.object = create_details_html(obs_id, point_data_df.iloc[0], plotter)
            except Exception as e:
                details_html_pane.object = f'<div class="alert alert-danger">Failed to load details: {e}</div>'
        pn.state.execute(load_details_html)

    def update_details_area(event, plotter):
        if not event or not event.new: return
        details_html_pane.object = '<div style="text-align:center; padding: 20px;"><h3>Loading...</h3></div>'
        details_area.visible = True
        async def load_details_html():
            try:
                obs_id = str(event.new['points'][0]['customdata'][0])
                point_data_df = plotter.hid_df[plotter.hid_df['ObsID'] == obs_id]
                if point_data_df.empty: raise ValueError(f"Could not find data for {obs_id}")
                details_html_pane.object = create_details_html(obs_id, point_data_df.iloc[0], plotter)
            except Exception as e:
                details_html_pane.object = f'<div class="alert alert-danger">Failed to load details: {e}</div>'
        pn.state.execute(load_details_html)

    def update_header(mission, card_to_update):
        card_to_update.title = f" Telescope: {mission.upper()}"


    def update_displayed_cards():
        """
        This is the new centralized function. It reads the current state from
        plotter_map and plot_scale_state, and redraws all cards.
        """

        cards = []
        for source_name, plotter in plotter_map.items():
            if plotter == "ERROR":
                cards.append(pn.Card(f"Could not load data for **{source_name}**.", css_classes=['plot-card-style'], width=480))
                continue

            current_scale = plot_scale_state.get(source_name, 'log')
            logger.info(f"--- Drawing '{source_name}' with y-axis scale: '{current_scale}'")
            try:
                plot_pane = plotter.hid_plot(yscale=current_scale)
            except TypeError:
                logger.warning("--- WARNING: plotter.hid_plot does not accept 'yscale'. Plot will use default scale.")
                plot_pane = plotter.hid_plot()
                if hasattr(plot_pane, 'object') and hasattr(plot_pane.object, 'update_yaxes'):
                    plot_pane.object.update_yaxes(type=current_scale)

            def plot_click_factory(p):
                return lambda event: update_float_panel_details(event, p)
            if isinstance(plot_pane, (pn.pane.Plotly, pn.pane.Matplotlib)):
             plot_pane.param.watch(plot_click_factory(plotter), 'click_data')

            linear_button = pn.widgets.Button(name='Linear Scale', button_type='default', width=120)
            log_button = pn.widgets.Button(name='Log Scale', button_type='default', width=120)
            
            linear_button.disabled = (current_scale == 'linear')
            log_button.disabled = (current_scale == 'log')

            def scale_change_factory(src, new_scale):
                def scale_click(event):
                    logger.info(f"--- [Button Click] State for '{src}' changed to '{new_scale}'. Triggering redraw.")
                    plot_scale_state[src] = new_scale
                    update_displayed_cards() # Redraw all cards with the new state
                return scale_click

            linear_button.on_click(scale_change_factory(source_name, 'linear'))
            log_button.on_click(scale_change_factory(source_name, 'log'))
            
            remove_button = pn.widgets.Button(name='Remove', button_type='danger', width=120)

            def remove_source_factory(src_name_to_remove):
                def remove_click(event):
                    if src_name_to_remove in plotter_map:
                        del plotter_map[src_name_to_remove]
                    if src_name_to_remove in plot_scale_state:
                        del plot_scale_state[src_name_to_remove]
                    update_displayed_cards() # Redraw without the removed card
                    details_area.visible = False
                return remove_click

            remove_button.on_click(remove_source_factory(source_name))

            button_controls = pn.Column(
                pn.Row(linear_button, log_button, align='center'),
                pn.Row(remove_button, align='center'),
                sizing_mode='stretch_width'
            )
            card = pn.Card(
                pn.pane.Markdown(f"### {source_name}", align='center'), plot_pane, button_controls,
                css_classes=['plot-card-style'], width=480
            )
            cards.append(card)

        plots_display_area.objects = cards if cards else [pn.pane.Markdown("### Select a source to add.", align='center')]
  

    def plot_local_hids_callback(event, selected_sources, hid_df=None):
      plot_type_heading.object = "## *Individual HID Diagrams*" 

      plots_display_area.loading = True
      details_area.visible = False
    
      if not selected_sources:
        plots_display_area.objects = [pn.pane.Alert("Please select at least one source to plot.", alert_type='warning')]
        plots_display_area.loading = False
        return

    # This loop now creates or updates a plotter for each selected source.
      for source_name in selected_sources:
        try:
            search_pattern = os.path.join(main_data_dir, '**', f"{source_name.replace(' ', '_')}.h5")
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                file_path = found_files[0]
                
                # This single command handles all cases:
                # - If hid_df is provided, it creates a plotter with the filtered data.
                # - If hid_df is None, it creates a plotter with the full data.
                # - If a plotter for 'source_name' already exists, it is replaced.
                plotter_map[source_name] = HIDPlotter(
                    h5_file_path=file_path,
                    name=source_name,
                    hid_df=hid_df 
                )
                plot_scale_state.setdefault(source_name, 'log')
            else:
                plotter_map[source_name] = "ERROR"
        except Exception as e:
            plotter_map[source_name] = "ERROR"
            logger.error(f"--> Error initializing plotter for {source_name}", exc_info=True)

      update_displayed_cards()
      plots_display_area.loading = False


    def plot_global_hid_callback(event, selected_sources):
        plot_type_heading.object = "# *Global HID Diagram*"
        plots_display_area.loading = True
        details_area.visible = False
        plots_display_area.objects = []  # Explicitly clear objects

        logger.info("Resetting plotters to ensure full data is loaded for the global plot.")
        for source_name in selected_sources:
          try:
             search_pattern = os.path.join(main_data_dir, '**', f"{source_name.replace(' ', '_')}.h5")
             found_files = glob.glob(search_pattern, recursive=True)
             if found_files:
                # Overwrite any existing plotter with a new one that has the FULL hid_df
                  plotter_map[source_name] = HIDPlotter(h5_file_path=found_files[0], name=source_name)
             else:
                  logger.warning(f"Could not find H5 file for {source_name} during plotter reset.")
          except Exception as e:
            logger.error(f"Failed to reset plotter for {source_name}: {e}")

        logger.info(f"Searching for H5 files in all subdirectories of: '{main_data_dir}'")
        global_df = get_global_hid_data(selected_sources, main_data_dir)
        logger.info(f"get_global_hid_data returned a DataFrame with {len(global_df)} rows.")

        if global_df.empty:
          alert_pane = pn.pane.Alert("Could not generate a combined plot. No data found.", alert_type='danger')
          plots_display_area.objects.append(alert_pane)
          plots_display_area.loading = False
          return

        logger.info("Calling create_global_hid_plot")
        plot_pane = create_global_hid_plot(global_df)
        
        if isinstance(plot_pane, pn.pane.Plotly):
            def handle_global_plot_click(event):
                if not event or not event.new: return
                try:
                    source_name = event.new['points'][0]['customdata'][1]
                    if source_name not in plotter_map:
                        search_pattern = os.path.join(main_data_dir, '**', f"{source_name.replace(' ', '_')}.h5")
                        found_files = glob.glob(search_pattern, recursive=True)
                        if found_files:
                            plotter_map[source_name] = HIDPlotter(h5_file_path=found_files[0], name=source_name)
                    plotter = plotter_map.get(source_name)
                    if plotter:
                       # update_details_area(event, plotter)
                        update_static_details_area(event, plotter)
                except (IndexError, KeyError) as e:
                    logger.error(f"Error processing global plot click: {e}")
            plot_pane.param.watch(handle_global_plot_click, 'click_data')

        combined_card = pn.Card(plot_pane, title="Combined Hardness-Intensity Diagram", css_classes=['plot-card-style'], sizing_mode='stretch_width')
        logger.info(f"Created combined card: type={type(combined_card)}, sizing_mode='{combined_card.sizing_mode}', width={combined_card.width}, height={combined_card.height}, visible={combined_card.visible}")
        plots_display_area.objects = [combined_card]
        logger.info(f"Updated plots_display_area.objects. New length: {len(plots_display_area.objects)}")
        logger.info(f"State of plots_display_area after update: type={type(plots_display_area)}, sizing_mode='{plots_display_area.sizing_mode}', width={plots_display_area.width}, height={plots_display_area.height}")

        plots_display_area.loading = False
        logger.info("LEAVING plot_global_hid_callback")
        logger.info("-" * 60)
    telescope_selector_widget.param.watch(lambda event: update_header(event.new, card_to_update=header_card), 'value')
    update_header(telescope_selector_widget.value, card_to_update=header_card)
    
    plots_display_area.objects = [pn.pane.Markdown("### Select one or more files and click a plot button.", align='center')]
    
    #plots_and_details = pn.Column(plots_display_area, details_area, sizing_mode='stretch_width')
    plots_and_details = pn.Column(plots_display_area, details_area, float_panel_placeholder, sizing_mode='stretch_width')

    return header_card, plots_and_details, plot_local_hids_callback, plot_global_hid_callback

