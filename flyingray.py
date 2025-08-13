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
.hamburger-menu {
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.bar {
    width: 25px;
    height: 3px;
    background-color: #333;
}
.nav-dropdown {
    display: none;
    position: absolute;
    top: 70px;
    right: 40px;
    background-color: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    padding: 10px;
    z-index: 1000;
}
.nav-dropdown.active {
    display: block;
}
.nav-link {
    display: block;
    color: #333;
    text-decoration: none;
    padding: 8px 12px;
}
.nav-link:hover {
    background-color: #f0f0f0;
}
"""
pn.config.raw_css.append(custom_css)


# --- Define Header HTML and JS ---
header_html = """
<div class="header-container">
    <div class="title-section">
        <h1>Flying ray</h1>
    </div>
    <div id="hamburger-icon" class="hamburger-menu">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
    </div>
    <div id="navigation-menu" class="nav-dropdown">
        <a href="#" class="nav-link">About</a>
    </div>
</div>
"""
header_js = """
<script>
    const hamburger = document.getElementById('hamburger-icon');
    const menu = document.getElementById('navigation-menu');
    if (hamburger && menu) {
        hamburger.addEventListener('click', () => {
            menu.classList.toggle('active');
        });
        document.addEventListener('click', (event) => {
            if (!menu.contains(event.target) && !hamburger.contains(event.target)) {
                menu.classList.remove('active');
            }
        });
    }
</script>
"""

"""
# --- Get all the UI components (CORRECTED SECTION) ---
status_pane = pn.pane.Markdown("Status: READY", min_height=100)

# 1. Create the pipeline controls. This function now returns two items.
# We only need the first one for the layout. The second is the plot area.
#pipeline_controls = create_pipeline_runner_tab(status_pane=status_pane)
pipeline_controls, plots_and_details_area = create_pipeline_runner_tab(status_pane=status_pane)

# 2. Extract the telescope selector widget from the controls we just created.
# It is the second item in the main column of the pipeline_controls.
# This is crucial for linking the two parts of the UI.
telescope_selector_widget = pipeline_controls[0]

# 3. Now create the plotting components, passing the correct widget.
# We don't need the callback functions here, so we use _ to ignore them.
plots_header, _, _, _ = create_h5_generator_tab(telescope_selector_widget=telescope_selector_widget)


# LEFT SIDE OF THE SCREEN
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

# RIGHT SIDE OF THE SCREEN (CORRECTED)
main_display_area = pn.Column(
    plots_header, # Use the correctly named variable
    pn.layout.Spacer(height=10),
    plots_and_details_area, # Use the plot area returned from pipeline.py
    sizing_mode='stretch_width'
)

# FINAL ASSEMBLY
main_content = pn.Row(
    left_column,
    pn.layout.Spacer(width=20),
    main_display_area,
    sizing_mode='stretch_width'
)

header_component = pn.pane.HTML(header_html + header_js, sizing_mode='stretch_width')

app_layout = pn.Column(
    header_component,
    main_content,
    sizing_mode='stretch_width',
    margin=(0, 20, 20, 20)
)

app_layout.servable()

# the orignal




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

header_component = pn.pane.HTML(header_html + header_js, sizing_mode='stretch_width')

app_layout = pn.Column(
    header_component,
    main_content,
    sizing_mode='stretch_width',
    margin=(0, 20, 20, 20)
)

app_layout.servable()


