
stingray_docs_link = "https://docs.stingray.science/en/stable/"
heasarc_pipeline_link = "https://github.com/matteobachetti/heasarc_retrieve_pipeline"

dashboard_info_text = f"""
<h1><em><strong>FlyingRay: An Interactive X-ray Binary Database</strong></em></h1>
<h2><em>A quick-look dashboard application that stores, analyzes, and organizes key data products from multiple observations of different telescopes. <em></h2>
<p><strong>FlyingRay</strong> removes the traditional complexities of mission-specific software, allowing you to generate and visualize key scientific products with a single click. The core of this project is powered by <strong><a href="{stingray_docs_link}" target="_blank">Stingray</a></strong>, the primary Python library for astronomical time-series analysis.</p>

<hr>

<h3><em><strong>Data Products</strong></em></h3>
<div>
    <div style="display: flex; gap: 20px; margin-bottom: 15px;">
        <div style="flex: 1; text-align: center;">
            <img src="assets/light_curve.png" alt="Example Light Curve" style="width: 90%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        <div style="flex: 1; text-align: center;">
            <img src="assets/pds.png" alt="Example Power Density Spectrum" style="width: 90%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
    </div>
    <div style="display: flex; gap: 20px; text-align: center;">
        <div style="flex: 1;">
            <h4>Light Curves</h4>
            <p>
Displays the photon count rate as a function of time. The plotting algorithm intelligently handles data gaps (GTIs); contiguous GTIs separated by less than one hour are rendered as a continuous panel. Significant gaps trigger a 'broken-axis' view. <strong>To maintain readability, the x-axis labels for each panel, <code>(a - b)</code>, show the truncated start and end times of that segment (typically the last five digits of the mission time).</strong> Vertical red lines within a panel mark smaller intervening data gaps.
</p>
        </div>
        <div style="flex: 1;">
            <h4>Power Density Spectra (PDS)</h4>
            <p>
Displays the fractional rms-squared normalized Power Density Spectrum in <em>Power × Frequency</em> units. The PDS is generated via <code>AveragedPowerspectrum</code> by averaging periodograms from multiple segments of the observation. The Poisson noise level, estimated from the high-frequency portion, is subtracted. The full-resolution PDS is shown in grey, with the logarithmically rebinned PDS overplotted in black to reveal the continuum shape and features like QPOs.
</p>
        </div>
    </div>
</div>

<hr>

<h3><em>This dashboard is built with the following open-source projects:</em></h3>