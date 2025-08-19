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
from PIL import Image
import copy

from src.dashboard.home.info import dashboard_info_text
from src.dashboard.plotting.functions import HIDPlotter, get_global_hid_data, create_global_hid_plot 
from src.dashboard.dataproducts.nicer_dataproducts import create_energy_band_lightcurves_nicer, create_pds_nicer
from src.dashboard.dataproducts.nustar_dataproducts import create_energy_band_lightcurves_nustar, create_pds_nustar
from src.dashboard.dataproducts.rxte_dataproducts import create_energy_band_lightcurves_rxte, create_pds_rxte


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MISSION = {
    "nustar": {
        "lc":  create_energy_band_lightcurves_nustar,
        "pds": create_pds_nustar,
        "soft_Eband": (3, 6),
        "high_Eband": (6, 10),
        "full_Eband": (3, 79),
    },
    "nicer": {
        "lc": create_energy_band_lightcurves_nicer,
        "pds": create_pds_nicer,
        "soft_Eband": (2, 4),
        "high_Eband": (4, 12),
        "full_Eband": (0.4, 12),
    },
    "rxte": {
    "lc": create_energy_band_lightcurves_rxte,
    "pds": create_pds_rxte,
    "soft_Eband": (3, 6),
    "high_Eband": (6, 15),
    "full_Eband": (3, 20),
    },
}


def save_plot_data_to_hdf5(group, plot_name, plot_data_bytes):
    if plot_data_bytes:# its is the png 
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

        
def calculate_hid_parameters(events, obsid, no_of_detectors, outburst_id, mission):
    try:
        total_exposure = np.sum(get_gti_lengths(events.gti))
        if total_exposure <= 0: return None
        intensity = len(events.filter_energy_range((MISSION[mission]["full_Eband"])).time) / total_exposure
        soft_rate = len(events.filter_energy_range((MISSION[mission]["soft_Eband"])).time) / total_exposure
        hard_rate = len(events.filter_energy_range((MISSION[mission]["high_Eband"])).time) / total_exposure
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


async def process_observation(evt_filepath, hdf5_file_path, outburst_id, mission):
    """
    Processes a single observation, including its outburst ID, and saves the data.
    """
    if not evt_filepath:
        logger.error("Received an empty list of event files to process.")
        return

    #obs_id_match = re.search(r'(\d{10,})', evt_filepath[0])
    obs_id_match = re.search(r'(\d{10,}|\d{5}-\d{2}-\d{2}-\d{2})', evt_filepath[0])
    if not obs_id_match:
        logger.error(f"Could not extract OBSID from event file: {evt_filepath[0]}")
        return
    obs_id = obs_id_match.group(1)
    try:
        lc_func, pds_func = None, None
        lc_func = MISSION[mission]["lc"]
        pds_func = MISSION[mission]["pds"]
        
        if mission == 'nustar' and len(evt_filepath) > 1:
            # Separate the event files for FPMA and FPMB the 2 detectors of nustar telescope
            file_a = next(f for f in evt_filepath if 'A_src1' in f)
            file_b = next(f for f in evt_filepath if 'B_src1' in f)
            ev_A = EventList.read(file_a, "hea")
            ev_B = EventList.read(file_b, "hea")
            lc_task = asyncio.to_thread(lc_func, ev_A, ev_B, obs_id)

            # For PDS and HID, i am joing the 2 evt files 
            combined_events = ev_A.join(ev_B)
            pds_task = asyncio.to_thread(pds_func, ev_A, ev_B, obs_id)
            hid_task = asyncio.to_thread(calculate_hid_parameters, combined_events, obs_id, 2, outburst_id, mission)
            

        elif mission == 'nicer':
            events_data = EventList.read(evt_filepath[0], "hea", additional_columns=["DET_ID"])
            ndet = len(set(events_data.det_id))
            lc_task = asyncio.to_thread(lc_func, events_data, obs_id)
            pds_task = asyncio.to_thread(pds_func, events_data, obs_id)
            hid_task = asyncio.to_thread(calculate_hid_parameters, events_data, obs_id, ndet, outburst_id, mission)

        elif mission == 'rxte':
            events_data = EventList.read(evt_filepath[0], "hea")
            ndet = 1  # Treat the PCA instrument as a single detector
            lc_task = asyncio.to_thread(lc_func, events_data, obs_id)
            pds_task = asyncio.to_thread(pds_func, events_data, obs_id)
            hid_task = asyncio.to_thread(calculate_hid_parameters, events_data, obs_id, ndet, outburst_id, mission)

        results = await asyncio.gather(lc_task, pds_task, hid_task)
        
        lc_plots, pds_plot, hid_params = results

        with h5py.File(hdf5_file_path, 'a') as hdf:
            if obs_id in hdf:
                del hdf[obs_id]
            obs_group = hdf.create_group(obs_id)
            
            if lc_plots:
                for name, data in lc_plots.items():
                    save_plot_data_to_hdf5(obs_group, name, data)
            else:
                print(f"--- DEBUG: WARNING! No light curve plots were generated for OBSID {obs_id}.")

            if pds_plot:
                save_plot_data_to_hdf5(obs_group, "pds", pds_plot)
            else:
                print(f"--- DEBUG: WARNING! No PDS plot was generated for OBSID {obs_id}.")
                
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

    except Exception as e:
        logger.error(f"Failed to process and save OBSID {obs_id}", exc_info=True)
        
        
def create_h5_generator_tab(telescope_selector_widget):
    plotter_map = {}
    plot_scale_state = {}
    card_map = {}
    plot_pane_map = {}
    plot_type_heading = pn.pane.Markdown("## *HOME*", margin=(15, 0, 0, 20)) 
    header_card = pn.Card(
        plot_type_heading,
        css_classes=['hid-card'],
        collapsible=True,
        sizing_mode='stretch_width'
    )

    plots_display_area = pn.FlexBox(
        sizing_mode='stretch_width',
        min_height=480, 
        justify_content='center',
        align_items='start'
    )
    
    float_panel_container = pn.Column(
        sizing_mode='stretch_width',
        visible= False
    ) 
   
    details_html_pane = pn.pane.HTML(sizing_mode='stretch_width')
    close_button = pn.widgets.Button(
        name='Close Details',
        button_type='primary'
    )
    
    details_area = pn.Column(
        details_html_pane,
        pn.Row(close_button, align='center'),
        visible=False,
        sizing_mode='stretch_width'
    )
    close_button.on_click(lambda e: setattr(details_area, 'visible', False))

    
    # Helper function to create HTML for any plot, avoiding repetition
def create_h5_generator_tab(telescope_selector_widget):
    plotter_map = {}
    plot_scale_state = {}
    card_map = {}
    plot_pane_map = {}
    plot_type_heading = pn.pane.Markdown("## *HOME*", margin=(15, 0, 0, 20)) 
    header_card = pn.Card(
        plot_type_heading,
        css_classes=['hid-card'],
        collapsible=True,
        sizing_mode='stretch_width'
    )

    plots_display_area = pn.FlexBox(
        sizing_mode='stretch_width',
        min_height=480, 
        justify_content='center',
        align_items='start'
    )
    
    float_panel_container = pn.Column(
        sizing_mode='stretch_width',
        visible= False
    ) 
   
    details_html_pane = pn.pane.HTML(sizing_mode='stretch_width')
    close_button = pn.widgets.Button(
        name='Close Details',
        button_type='primary'
    )
    
    details_area = pn.Column(
        details_html_pane,
        pn.Row(close_button, align='center'),
        visible=False,
        sizing_mode='stretch_width'
    )
    close_button.on_click(lambda e: setattr(details_area, 'visible', False))

    def create_details_html(obs_id, selected_row, plotter):

      style = "width: 400px; margin: 10px; text-align: center;"
      pds_style = "width: 400px; margin: 10px auto; text-align: center;"

    # 1. Get all available plot data first
      all_lc_plots = plotter.get_all_lightcurve_pngs(plotter.h5_file_path, obs_id)
      pds_png_data = plotter.get_pds_png(plotter.h5_file_path, obs_id)

      pds_html = ""
      if pds_png_data is not None:
        encoded_pds_image = base64.b64encode(pds_png_data).decode('utf-8')
        pds_img_src = f'data:image/png;base64,{encoded_pds_image}'
        pds_html = f'<div style="{pds_style}"><h4>Power Density Spectrum</h4><img src="{pds_img_src}" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>'
        
      else:
        pds_html = f'<div style="{pds_style}"><h4>Power Density Spectrum</h4><p>Not available</p></div>'
    
# --- 3. THE CONDITIONAL LAYOUT LOGIC ---
    
    # First, convert all plot data to a list of HTML strings
      lc_html_list = []
      for title, png_data in all_lc_plots.items():
        encoded_image = base64.b64encode(png_data).decode('utf-8')
        img_src = f'data:image/png;base64,{encoded_image}'
        lc_html_list.append(f'<div style="{style}"><h4>Light Curve ({title})</h4><img src="{img_src}" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>')

    # Now, arrange the plots using the list we just made
      plots_html_block = ""
      num_lcs = len(lc_html_list)

      if num_lcs == 3:
        # Case for 3 LCs -> Perfectly aligned 2x2 grid
        # 1. Combine all four plot HTML strings
        all_plots_html = lc_html_list[0] + lc_html_list[1] + lc_html_list[2] + pds_html
        
        # 2. Put them in a single wrapping container with a fixed width
        # The max-width (e.g., 850px) is crucial for forcing a 2-column wrap
        plots_html_block = f"""
            <div style="display: flex; flex-wrap: wrap; justify-content: center; max-width: 850px; margin: auto;">
                {all_plots_html}
            </div>
        """
      elif num_lcs == 2:
        # Case for 2 LCs -> LCs side-by-side, PDS below
        all_lcs_html = "".join(lc_html_list)
        plots_html_block = f"""
            <div style="display: flex; justify-content: center; flex-wrap: wrap;">{all_lcs_html}</div>
            <div>{pds_html}</div>
        """
      else:
        # Case for 0 or 1 LC -> All side-by-side
        lc_html = lc_html_list[0] if lc_html_list else f'<div style="{style}"><h4>Light Curve</h4><p>Not available</p></div>'
        plots_html_block = f'<div style="display: flex; justify-content: center; flex-wrap: wrap;">{lc_html}{pds_html}</div>'

    # --- 4. Get text data (unchanged) ---
      intensity_val = selected_row.get('Intensity', np.nan)
      normalized_intensity_val = selected_row.get('Normalized_Intensity', np.nan)
      detectors_val = selected_row.get('No_of_detectors', np.nan)
      outburst_val = selected_row.get('Outburst', np.nan)


    # --- 5. Assemble the final HTML with the dynamically created plot block ---
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
        <div style="text-align: center;">
            {plots_html_block}
        </div>
    </div>"""
      return final_html

    def update_float_panel_details(event, plotter):
     try:
        obs_id = str(event.new['points'][0]['customdata'][0])
        
        # 1. Get all plot data first
        all_lc_plots = plotter.get_all_lightcurve_pngs(plotter.h5_file_path, obs_id)
        pds_png_data_np = plotter.get_pds_png(plotter.h5_file_path, obs_id)

        # 2. Create the Panel Panes for all plots
        lc_panes = []
        if not all_lc_plots:
            lc_panes.append(pn.pane.Markdown("### Lightcurve not available.", width=400))
        else:
            for title, png_data_np in all_lc_plots.items():
                png_bytes = png_data_np.tobytes()
                lc_panes.append(pn.pane.PNG(png_bytes, width=400, name=title))

        if pds_png_data_np is not None:
            pds_bytes = pds_png_data_np.tobytes()
            # Default to 400px width for the side-by-side case
            pds_pane = pn.pane.PNG(pds_bytes, width=400, name="PDS")
        else:
            pds_pane = pn.pane.Markdown("### PDS not available.", width=400)
        
        # --- 3. THE CONDITIONAL LAYOUT LOGIC ---
        #title_pane = pn.pane.Markdown(f"### Details for ObsID: {obs_id}", align='center')
        content_layout = None
        if len(all_lc_plots) == 3:
            # First row has the first two LCs
            row1 = pn.Row(lc_panes[0], lc_panes[1], align='center')
            # Second row has the third LC and the PDS
            row2 = pn.Row(lc_panes[2], pds_pane, align='center')
            # Combine the rows in a column
            content_layout = pn.Column(row1, row2)
            
        elif len(all_lc_plots) == 2:
            lc_row = pn.Row(*lc_panes, align='center')
            pds_row = pn.Row(pds_pane, align='center')
            content_layout = pn.Column(lc_row, pds_row)
                
        else:
            # --- Case 2: One (or zero) LCs -> All plots in a single side-by-side row. ---
            all_panes = lc_panes + [pds_pane]
            content_layout = pn.Column( pn.Row(*all_panes, align='center'))
            
        # 4. Create and show the FloatPanel
        new_float_panel = pn.layout.FloatPanel(
            content_layout, 
            name=f"*{telescope_selector_widget.value}* {plotter.name} - {obs_id}",
            contained=False,
            position='center-top',
            status='normalized',
           # theme="#f2f2f2",
            theme='#e9e9e9',
            width=850, 
            height=400,
        )
        float_panel_container.append(new_float_panel)

     except Exception as e:
        logger.error(f"Failed to show FloatPanel details: {e}", exc_info=True)

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

   # def update_header(mission, card_to_update):
    #    card_to_update.title = f" Telescope: {mission.upper()}"

    def update_displayed_cards():
    # --- 2. Create and add new cards that are not yet displayed ---
      current_sources = set(plotter_map.keys())
      existing_cards = set(card_map.keys())
      for source_to_remove in existing_cards - current_sources:
        card_map.pop(source_to_remove, None)
        plot_pane_map.pop(source_to_remove, None)

      new_cards_list = [] 
          
        
      for unique_key, plotter in plotter_map.items():
          source_name, mission = unique_key
          if plotter == "ERROR":
              card = pn.Card(f"Could not load data for **{source_name}**.", css_classes=['plot-card-style'], width=480)
              new_cards_list.append(card)
              card_map[source_name] = card
              new_cards_list.append(card)
              continue

        # Initial plot creation
          current_scale = plot_scale_state.get(unique_key, 'log')
          plot_pane = plotter.hid_plot(yscale=current_scale)
          plot_pane_map[unique_key] = plot_pane

          def plot_click_factory(p):
            return lambda event: update_float_panel_details(event, p)
        
          if isinstance(plot_pane, (pn.pane.Plotly, pn.pane.Matplotlib)):
              plot_pane.param.watch(plot_click_factory(plotter), 'click_data')

          linear_button = pn.widgets.Button(
              name='Linear Scale', 
              button_type='default',
              width=120
          )
          log_button = pn.widgets.Button(
              name='Log Scale',
              button_type='default', 
              width=120
          )

          linear_button.disabled = (current_scale == 'linear')
          log_button.disabled = (current_scale == 'log')

          def scale_change_factory(src_name, new_scale, lin_btn, log_btn):
              def scale_click(event):
                  logger.info(f"--- [Targeted Update] State for '{src_name}' changed to '{new_scale}'.")
                  # 1. Update the state
                  plot_scale_state[src_name] = new_scale
                  pane_to_update = plot_pane_map[src_name]

                # 2. Re-generate ONLY the plot for this source
                  new_figure = plotter_map[src_name].hid_plot(yscale=new_scale).object
                  pane_to_update.object = new_figure

                # 4. Update the button states on THIS CARD ONLY
                  lin_btn.disabled = (new_scale == 'linear')
                  log_btn.disabled = (new_scale == 'log')
              return scale_click

          linear_button.on_click(scale_change_factory(unique_key, 'linear', linear_button, log_button))
          log_button.on_click(scale_change_factory(unique_key, 'log', linear_button, log_button))

          remove_button = pn.widgets.Button(name='Remove', button_type='danger', width=120)

          def remove_source_factory(src_name_to_remove):
              def remove_click(event):
                  plotter_map.pop(src_name_to_remove, None)
                  plot_scale_state.pop(src_name_to_remove, None)
                  update_displayed_cards()  # Call this to remove the card from the display
              return remove_click
        
          remove_button.on_click(remove_source_factory(unique_key))

          button_controls = pn.Column(
            pn.Row(linear_button, log_button, align='center'),
            pn.Row(remove_button, align='center'),
            sizing_mode='stretch_width'
          )
        
          card = pn.Card( 
                  plot_pane,
                  button_controls,
                  header = f"### {source_name} ({mission.upper()})",
                  collapsible=False,           
                  css_classes=['plot-card-style'], 
                  width=480
         )


          card_map[unique_key] = card  # Store reference to the whole card
          new_cards_list.append(card)

    # --- 3. Update the final display area ---
      if not card_map:
          plots_display_area.objects = [pn.pane.Markdown("### Select a source to add.", align='center')]
      else:
          plots_display_area.objects = new_cards_list
    
    def plot_local_hids_callback(event, selected_sources, mission, hid_df=None):
      plot_type_heading.object = "## *Individual HID Diagrams*" 

      plots_display_area.loading = True
      details_area.visible = False
    
      if not selected_sources:
        plots_display_area.objects = [pn.pane.Alert("Please select at least one source to plot.", alert_type='warning')]
        plots_display_area.loading = False
        return

    # This loop now creates or updates a plotter for each selected source.
      for source_name in selected_sources:
        unique_key = (source_name, mission)
        try:
            search_pattern = os.path.join("data", mission, '**', f"{source_name.replace(' ', '_')}.h5")
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                file_path = found_files[0]
                
                # This single command handles all cases:
                # - If hid_df is provided, it creates a plotter with the filtered data.
                # - If hid_df is None, it creates a plotter with the full data.
                # - If a plotter for 'source_name' already exists, it is replaced.
                plotter_map[unique_key] = HIDPlotter(
                    h5_file_path=file_path,
                    name=source_name,
                    hid_df=hid_df 
                )
                plot_scale_state.setdefault(unique_key, 'log')
            else:
                plotter_map[unique_key] = "ERROR"     
        except Exception as e:
            plotter_map[unique_key] = "ERROR"          
            logger.error(f"--> Error initializing plotter for {source_name}", exc_info=True)

      update_displayed_cards()
      plots_display_area.loading = False


    def plot_global_hid_callback(event, selected_sources, mission):
        plot_type_heading.object = "## *Global HID Diagram*"
        plots_display_area.loading = True
        details_area.visible = False
        plots_display_area.objects = []  # Explicitly clear objects

        logger.info("Resetting plotters to ensure full data is loaded for the global plot.")
        for source_name in selected_sources:
          try:
             search_pattern = os.path.join("data", mission, '**', f"{source_name.replace(' ', '_')}.h5")
             found_files = glob.glob(search_pattern, recursive=True)
             if found_files:
                 unique_key = (source_name, mission)
                # Overwrite any existing plotter with a new one that has the FULL hid_df
                 plotter_map[unique_key] = HIDPlotter(h5_file_path=found_files[0], name=source_name)
             else:
                  logger.warning(f"Could not find H5 file for {source_name} during plotter reset.")
          except Exception as e:
            logger.error(f"Failed to reset plotter for {source_name}: {e}")


        global_df = get_global_hid_data(selected_sources, "data", mission)

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
                        search_pattern = os.path.join("data", mission, '**', f"{source_name.replace(' ', '_')}.h5")
                        found_files = glob.glob(search_pattern, recursive=True)
                        if found_files:
                            plotter_map[source_name] = HIDPlotter(h5_file_path=found_files[0], name=source_name)
                    plotter = plotter_map.get(source_name)
                    if plotter:
                        update_details_area(event, plotter)
                except (IndexError, KeyError) as e:
                    logger.error(f"Error processing global plot click: {e}")
            plot_pane.param.watch(handle_global_plot_click, 'click_data')

        combined_card = pn.Card(
            plot_pane,
            title="Combined Hardness-Intensity Diagram",
            css_classes=['plot-card-style'], 
            sizing_mode='stretch_width'
        )
        
        plots_display_area.objects = [combined_card]
        plots_display_area.loading = False

    #telescope_selector_widget.param.watch(lambda event: update_header(event.new, card_to_update=header_card), 'value')
    #update_header(telescope_selector_widget.value, card_to_update=header_card)
    
   # plots_display_area.objects = [pn.pane.Markdown("### Select one or more files and click a plot button.", align='center')]
    info_pane = pn.pane.Markdown(dashboard_info_text, sizing_mode='stretch_width')
    plots_display_area.objects = [info_pane]
    
    plots_and_details = pn.Column(
        plots_display_area,
        details_area, 
        float_panel_container, 
        sizing_mode='stretch_width'
    )

    return header_card, plots_and_details, plot_local_hids_callback, plot_global_hid_callback