import logging
import copy
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from stingray import Lightcurve, AveragedPowerspectrum, EventList
from stingray.gti import get_gti_lengths

# Set up a logger for this file
logger = logging.getLogger(__name__)

def create_energy_band_lightcurves(events, obs_id, dt=10, large_gap_threshold=3600):
    """
    Generates professional light curve plots, automatically choosing the best
    visualization method and adding a diagnostic note for high variability.
    """
    plots_data = {}
    variability_note = None # Renamed for clarity
    energy_bands = {'0.4-12_keV': (0.4, 12)}

    for band_name, (e_min, e_max) in energy_bands.items():
        try:
            # --- Data Preparation ---
            energy_mask = (events.energy >= e_min) & (events.energy <= e_max)
            filtered_times = events.time[energy_mask]

            if len(filtered_times) == 0:
                logging.warning(f"No events in band {band_name} for {obs_id}. Skipping.")
                continue

            lc_full = Lightcurve.make_lightcurve(filtered_times, dt=dt, gti=events.gti, mjdref=events.mjdref)
            lc = copy.copy(lc_full)
            lc.apply_gtis()

            if lc.time is None or len(lc.time) == 0:
                logging.warning(f"No data left after applying GTIs for {obs_id} band {band_name}. Skipping.")
                continue

            # --- Calculate intensity ratio note ---
            valid_counts = lc.countrate[np.isfinite(lc.countrate) & (lc.countrate > 0)]
            if len(valid_counts) > 1:
                min_intensity = np.min(valid_counts)
                max_intensity = np.max(valid_counts)
                if min_intensity > 0:
                    ratio = max_intensity / min_intensity
                    if ratio > 50:
                        variability_note = "NOTE: High variability detected (Max/Min > 50). Consider custom processing flags."

            # --- Plotting Rules ---
            all_gtis = sorted([tuple(g) for g in lc.gti], key=lambda x: x[0]) if lc.gti is not None else []
            plots_to_generate = {}
            if len(all_gtis) <= 2 or len(all_gtis) > 10:
                 plots_to_generate['full_continuous_lc'] = {'title': f"Light Curve {band_name}: {obs_id} (dt={dt}s)", 'plot_type': 'simple_continuous'}
            else:
                gti_segments = []
                # Handle edge case of no GTIs
                if not all_gtis:
                    return {}, None
                current_segment = [all_gtis[0]]
                for i in range(len(all_gtis) - 1):
                    gap = all_gtis[i+1][0] - all_gtis[i][1]
                    if gap > large_gap_threshold:
                        gti_segments.append(current_segment)
                        current_segment = [all_gtis[i+1]]
                    else:
                        current_segment.append(all_gtis[i+1])
                gti_segments.append(current_segment)
                num_large_gaps = len(gti_segments) - 1

                if 4 <= num_large_gaps <= 7:
                    split_index = (len(gti_segments) + 1) // 2
                    plots_to_generate['part1'] = {
                        'segments': gti_segments[:split_index],
                        'title': f"Light Curve Part 1 {band_name}: {obs_id} (dt={dt}s)", 
                        'plot_type': 'broken_axis'
                    }
                    plots_to_generate['part2'] = {
                        'segments': gti_segments[split_index:],
                        'title': f"Light Curve Part 2 {band_name}: {obs_id} (dt={dt}s)", 
                        'plot_type': 'broken_axis'
                    }
                elif num_large_gaps in [8, 9]:
                    n = len(gti_segments)
                    idx1, idx2 = n // 3, 2 * n // 3
                    plots_to_generate['part1'] = {
                        'segments': gti_segments[:idx1], 
                        'title': f"Light Curve Part 1 {band_name}: {obs_id} (dt={dt}s)",
                        'plot_type': 'broken_axis'
                    }
                    plots_to_generate['part2'] = {
                        'segments': gti_segments[idx1:idx2],
                        'title': f"Light Curve Part 2 {band_name}: {obs_id} (dt={dt}s)",
                        'plot_type': 'broken_axis'
                    }
                    plots_to_generate['part3'] = {
                        'segments': gti_segments[idx2:], 
                        'title': f"Light Curve Part 3 {band_name}: {obs_id} (dt={dt}s)", 
                        'plot_type': 'broken_axis'
                    }
                else:
                    plots_to_generate['full_light_curve'] = {
                        'segments': gti_segments, 
                        'title': f"Light Curve {band_name}: {obs_id} (dt={dt}s)", 
                        'plot_type': 'broken_axis'
                    }

            # --- Plotting Loop ---
            for plot_key, plot_info in plots_to_generate.items():
                title = plot_info['title']

                if plot_info['plot_type'] == 'simple_continuous':
                    fig, ax = plt.subplots()
                    ax.plot(lc.time, lc.countrate, color='k', marker='o', markersize=2, linestyle='-', linewidth=0.1)
                    if lc_full.gti is not None and len(lc_full.gti) > 1:
                        for i in range(len(lc_full.gti) - 1):
                            ax.axvspan(lc_full.gti[i, 1], lc_full.gti[i + 1, 0], alpha=0.3, color='red', zorder=0)
                    if len(lc_full.time) > 1:
                        ax.set_xlim(lc_full.time[0], lc_full.time[-1])
                    ax.set_title(title)
                    ax.set_ylabel(r'Counts s$^{-1}$')
                    ax.set_xlabel(r'Time [s]')
                else: # broken_axis
                    segments_to_plot = plot_info['segments']
                    populated_segments = [s for s in segments_to_plot if s and np.any((lc.time >= s[0][0]) & (lc.time <= s[-1][1]))]
                    if not populated_segments: continue

                    num_axes = len(populated_segments)
                    #fig_width = 4 * num_axes if num_axes > 1 else 8
                    fig, axes = plt.subplots(1, num_axes, sharey=True)
                    axes = np.atleast_1d(axes)

                    for i, segment_gtis in enumerate(populated_segments):
                        ax = axes[i]
                        start_time, end_time = segment_gtis[0][0], segment_gtis[-1][1]
                        mask = (lc.time >= start_time) & (lc.time <= end_time)
                        if not np.any(mask): continue
                        ax.plot(lc.time[mask], lc.countrate[mask], 'k-o', markersize=2, linewidth=0.1)
                        
                        if len(segment_gtis) > 1:
                            for j in range(len(segment_gtis) - 1):
                                mid = (segment_gtis[j][1] + segment_gtis[j+1][0]) / 2.0
                                ax.axvline(mid, color='red', linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)

                        ax.tick_params(direction='in', top=True, right=False)
                        ax.set_xticks([])
                        ax.yaxis.grid(False)
                        
                        full_label = f"{int(start_time % 100000)} - {int(end_time % 100000)}"
                        ax.set_xlabel(full_label, fontsize=9)
                        
                        if num_axes > 1:
                            if i == 0: ax.spines['right'].set_visible(False)
                            elif i == num_axes - 1:
                                ax.spines['left'].set_visible(False);
                                ax.tick_params(labelleft=False, left=False)
                            else:
                                ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False);
                                ax.tick_params(labelleft=False, left=False)

                        if variability_note and i == 0:
                            ax.text(0.9, 0.95, variability_note,
                                    transform=ax.transAxes, ha='right', va='top',
                                    fontsize=9, color='red', style='italic',
                                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 4})

                    if num_axes > 1:
                        d = .015; kwargs = dict(transform=fig.transFigure, color='k', clip_on=False, lw=1)
                        for k in range(num_axes - 1):
                            p1, p2 = axes[k].get_position(), axes[k+1].get_position()
                            x_mid = (p1.x1 + p2.x0) / 2
                            fig.add_artist(plt.Line2D([x_mid - d, x_mid + d], [p1.y0 - d, p1.y0 + d], **kwargs))
                            fig.add_artist(plt.Line2D([x_mid - d, x_mid + d], [p1.y1 - d, p1.y1 + d], **kwargs))
                    
                    ymin, ymax = axes[0].get_ylim()
                    if ymax > 0: axes[0].set_ylim(bottom= -0.15 * ymax, top=ymax * 1.15)
                    
                    axes[0].set_ylabel(r'Counts s$^{-1}$')
                    fig.supxlabel('Time [s]')
                    fig.suptitle(title)

                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plots_data[f"lightcurve_{band_name}_{plot_key}"] = buf.getvalue()
                plt.close(fig)

        except Exception as e:
            logging.error(f"Failed to create lightcurve for {obs_id} band {band_name}", exc_info=True)
            return {}, None

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

        