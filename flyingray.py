import panel as pn

from src.dashboard.HDF5.h5 import create_h5_generator_tab
from src.dashboard.pipeline.controls import create_pipeline_runner_tab

print(f"Panel version being used: {pn.__version__}")

# --- Set up extensions and CSS ---
#pn.extension('plotly')
pn.extension('plotly')
pn.extension('floatpanel')
custom_css = """
body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.hid-card {
  border: 1px solid #ddd !important;
  border-radius: 8px !important;
  box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important;
  padding: 15px !important;
  background: #FFFFFF !important;
}
.plot-card-style {
    border: 1px solid #ccc !important;
    padding: 10px !important;
    border-radius: 8px !important;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important;
    background: #ffffff !important;
    margin: 10px;
}
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 40px;
    background-color: #ffffff;
    border-bottom: 1px solid #e0e0e0;
}
.title-section h1 {
    margin: 0;
    font-family: 'Georgia', 'Times New Roman', Times, serif;
    font-size: 2.1em;
    font-weight: 500;
    color: #333;
}

"""
pn.config.raw_css.append(custom_css)

header_html = """
<div class="header-container">
    <div class="title-section">
        <h1>Flyingray</h1>
    </div>
    <div class="title-section">
        <h1>
            ( The Interactive X-ray Binaries Database ) Built with  
            <a href="https://github.com/StingraySoftware" 
               style="color: green; text-decoration: none;" 
               target="_blank">StingraySoftware</a>
        </h1>
    </div>
</div>
"""
tels_name = pn.widgets.Select(name="Select Telescope", options=['nicer', 'nustar', 'rxte'], sizing_mode='stretch_width', height=60)
status_pane = pn.pane.Markdown("Status: READY", min_height=100, sizing_mode='stretch_width')

(
    plots_header,
    plots_and_details,
    plot_local_hids_callback,
    plot_global_hid_callback,
) = create_h5_generator_tab(telescope_selector_widget=tels_name) # FIX: Use the correct variable name 'tels_name'
# 4. Assemble the final layout.

# 3. Create the Pipeline controls, passing in the shared widgets and callbacks.
pipeline_controls, tels_name = create_pipeline_runner_tab(
    status_pane=status_pane,
    plot_local_hids_callback=plot_local_hids_callback,
    plot_global_hid_callback=plot_global_hid_callback,
)

# LEFT SIDE
left_column = pn.Column(
    pn.Card(
        pipeline_controls,
        title="Pipeline & Plotting Controls",
        css_classes=['hid-card'],
        width=600
    ),
    pn.layout.Spacer(height=20),
    pn.Card(
        status_pane,
        title="Pipeline Status",
        css_classes=['hid-card'],
        width=600
    ),
)

# RIGHT SIDE
right_column = pn.Column(
    plots_header,
    pn.layout.Spacer(height=10),
    plots_and_details,
    sizing_mode='stretch_width'
)

# FINAL ASSEMBLY
main_content = pn.Row(
    left_column,
    pn.layout.Spacer(width=20),
    right_column,
    sizing_mode='stretch_width'
)

header_component = pn.pane.HTML(header_html, sizing_mode='stretch_width')

app_layout = pn.Column(
    header_component,
    pn.layout.Spacer(height=5),
    main_content,
    sizing_mode='stretch_width',
    margin=(0, 20, 20, 20)
)

app_layout.servable()
