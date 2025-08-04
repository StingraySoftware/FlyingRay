

import panel as pn
from pipeline import create_pipeline_runner_tab
from h5 import create_h5_generator_tab

pn.extension('plotly')

# 1. Defines all custom CSS, including the new .main-area style
custom_css = """
body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
/* --- ADDED: Style for the new outer container for Step 1 --- */
.main-area {
    border: 4px dashed #dcdcdc !important;
    border-radius: 8px;
    padding: 10px;
    background-color: #ffffff;
}
.plots-area {
    border: 4px dashed #dcdcdc !important;
    border-radius: 8px;
    padding: 10px;
    min-height: 520px;
    background-color: #ffffff;
    overflow-y: auto;
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
.remove-button-style button {
    background-color: #f44336 !important;
    color: white !important;
    border: none !important;
    padding: 5px 10px !important;
    border-radius: 3px !important;
    cursor: pointer !important;
    margin-top: 5px !important;
    font-weight: bold !important;
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
.year-text {
    font-weight: 300;
    color: #777;
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
# Inject the custom CSS into the page
pn.config.raw_css.append(custom_css)


# 2. Defines the HTML for the header
header_html = """
<div class="header-container">
    <div class="title-section">
        <h1>My Astro Pipeline. <span class="year-text">2025</span></h1>
    </div>
    <div id="hamburger-icon" class="hamburger-menu">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
    </div>
    <div id="navigation-menu" class="nav-dropdown">
        <a href="#pipeline-runner" class="nav-link">Pipeline Runner</a>
        <a href="#h5-generator" class="nav-link">H5 Generator</a>
        <a href="#about" class="nav-link">About</a>
    </div>
</div>
"""

# 3. Defines the JavaScript to make the hamburger menu interactive
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

# 4. Creates UI components
status_pane = pn.pane.Markdown("Status: Idle")
pipeline_runner_section = create_pipeline_runner_tab(status_pane=status_pane)
h5_generator_section = create_h5_generator_tab()


# 5. Assembles the final page layout
# --- MODIFIED: This section now wraps the Step 1 cards in an outer container ---

# First, create the two separate cards for Step 1
pipeline_container = pn.Card(
    pipeline_runner_section,
    title="Step 1: Run Data Processing Pipeline",
    css_classes=['hid-card'],
    width=800
    
)
status_card = pn.Card(
    status_pane,
    title="Pipeline Status",
    css_classes=['hid-card'],
    width=400,
    height=700,
    
)

# Second, create the new outer wrapper to contain both cards
step_1_wrapper = pn.Row(
    pipeline_container,
    pn.layout.Spacer(width=50),
    status_card,
    pn.layout.Spacer(width=80),
    css_classes=['main-area'],
    sizing_mode='stretch_width'
)

# Finally, assemble the main page content
main_content = pn.Column(
    step_1_wrapper,
    pn.layout.Divider(),
    h5_generator_section,
    sizing_mode='stretch_width',
)


# 6. Creates the final servable app layout
header_component = pn.pane.HTML(header_html + header_js, sizing_mode='stretch_width')

app_layout = pn.Column(
    header_component,
    main_content,
    sizing_mode='stretch_width',
    margin=(0, 20, 20, 20)
)

app_layout.servable()


