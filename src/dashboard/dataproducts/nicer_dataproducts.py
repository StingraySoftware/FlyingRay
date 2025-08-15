import logging
import copy
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from stingray import Lightcurve, AveragedPowerspectrum, EventList
from stingray.gti import get_gti_lengths

logger = logging.getLogger(__name__)

def create_energy_band_lightcurves_nicer(events, obs_id, dt=10):
    """
    Generates professional light curve plots, automatically choosing the best
    visualization method based on the total number of GTIs in the data.
    """
    large_gap_threshold=3600
    plots_data = {}
    energy_bands = {'0.4-12_keV': (0.4, 12)}

    for band_name, (e_min, e_max) in energy_bands.items():
        try:
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

            note_to_add = None
            # Filter for valid, positive count rates to avoid errors
            valid_counts = lc.countrate[np.isfinite(lc.countrate) & (lc.countrate > 0)]
            if len(valid_counts) > 1:
                min_intensity = np.min(valid_counts)
                max_intensity = np.max(valid_counts)
                # Check for non-zero min intensity to prevent division by zero
                if min_intensity > 0:
                    ratio = max_intensity / min_intensity
                    if ratio > 50:
                        note_to_add = "NOTE:Instrumeental noice detected. Consider custom processing flags.\n (Max. Intensity/Min Intensity)>50"

            all_gtis = sorted([tuple(g) for g in lc.gti], key=lambda x: x[0]) if lc.gti is not None else []
            plots_to_generate = {}

            # RULE 1: For simple (<=2 GTIs) or very complex (>9 GTIs) data, use a single continuous plot.
            # --- MODIFIED LOGIC: Changed from >5 to >9 to accommodate the new rules. ---
            if len(all_gtis) <= 2 or len(all_gtis) > 10: # Changed from >9 to >10 to include 9 gaps case
                plots_to_generate['full_continuous_lc'] = {
                    'title': f"Light Curve {band_name}: {obs_id} (dt={dt}s)",
                    'plot_type': 'simple_continuous'
                }
            else:
                gti_segments = []
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
                

                # CASE A: 4 to 7 gaps -> Split into 2 plots
                if 4 <= num_large_gaps <= 7:
                    split_index = (len(gti_segments) + 1) // 2
                    plots_to_generate['part1'] = {'segments': gti_segments[:split_index], 'title': f"Light Curve Part 1 {band_name}: {obs_id} (dt={dt}s)", 'plot_type': 'broken_axis'}
                    plots_to_generate['part2'] = {'segments': gti_segments[split_index:], 'title': f"Light Curve Part 2 {band_name}: {obs_id} (dt={dt}s)", 'plot_type': 'broken_axis'}

                # CASE B: 8 or 9 gaps -> Split into 3 plots
                elif num_large_gaps in [8, 9]:
                    n = len(gti_segments)
                    # Calculate split points for three nearly equal parts
                    idx1 = n // 3
                    idx2 = 2 * n // 3
                    plots_to_generate['part1'] = {'segments': gti_segments[:idx1], 'title': f"Light Curve: {obs_id} (Part 1)", 'plot_type': 'broken_axis'}
                    plots_to_generate['part2'] = {'segments': gti_segments[idx1:idx2], 'title': f"Light Curve: {obs_id} (Part 2)", 'plot_type': 'broken_axis'}
                    plots_to_generate['part3'] = {'segments': gti_segments[idx2:], 'title': f"Light Curve: {obs_id} (Part 3)", 'plot_type': 'broken_axis'}
                
                # CASE C: fewer than 4 gaps -> create one single broken-axis plot
                else:
                    plots_to_generate['full_light_curve'] = {'segments': gti_segments, 'title': f"Light Curve: {obs_id}", 'plot_type': 'broken_axis'}

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

                    populated_segments = []
                    for segment in segments_to_plot:
                        start_time, end_time = segment[0][0], segment[-1][1]
                        mask = (lc.time >= start_time) & (lc.time <= end_time)
                        if np.any(mask) and np.any(np.isfinite(lc.countrate[mask])):
                            populated_segments.append(segment)
                    
                    if not populated_segments: continue

                    num_axes = len(populated_segments)
                    fig, axes = plt.subplots(1, num_axes, sharey=True)                 
                    axes = np.array(axes).flatten()

                    for i, segment_gtis in enumerate(populated_segments):
                      ax = axes[i]
                      start_time = segment_gtis[0][0]
                      end_time = segment_gtis[-1][1]
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
                                                
                      # 1. Determine the number of digits to show (e.g., last 4).
                      divisor = 100000
                      # 2. Calculate the "last digits" for the start and end times.
                      last_digits_start = int(start_time) % divisor
                      last_digits_end = int(end_time) % divisor
                      # 3. Calculate the base value (the part of the number we are hiding).
                      offset = int(start_time) - last_digits_start
                      
                      # 4. Format the two lines of the label.
                      time_range_label = f"{last_digits_start} - {last_digits_end}"
                      offset_label = f"(+{offset:.3e} s)"
                      
                      # 5. Combine and set the label for this specific panel.
                      full_label = f"{time_range_label}"
                      ax.set_xlabel(full_label, fontsize=9)
                      
                      if num_axes > 1:
                          if i == 0: ax.spines['right'].set_visible(False)
                          elif i == num_axes - 1:
                              ax.spines['left'].set_visible(False)
                              ax.tick_params(labelleft=False, left=False)
                          else:
                              ax.spines['left'].set_visible(False)
                              ax.spines['right'].set_visible(False)
                              ax.tick_params(labelleft=False, left=False)
                    
                    if num_axes > 1:
                     d = .015
                     kwargs = dict(transform=fig.transFigure, color='k', clip_on=False, lw=1)
                     for i in range(num_axes - 1):
                         p1, p2 = axes[i].get_position(), axes[i+1].get_position()
                         x_mid = (p1.x1 + p2.x0) / 2
                         fig.add_artist(plt.Line2D([x_mid - d, x_mid + d], [p1.y0 - d, p1.y0 + d], **kwargs))
                         fig.add_artist(plt.Line2D([x_mid - d, x_mid + d], [p1.y1 - d, p1.y1 + d], **kwargs))

                    ymin, ymax = axes[0].get_ylim()
                    if ymax > 0:
                        axes[0].set_ylim(bottom= -0.15 * ymax, top=ymax * 1.15)

                    
                    axes[0].set_ylabel(r'Counts s$^{-1}$')                    
                    fig.supxlabel('Time [s]')
                    fig.suptitle(title)    

                if note_to_add:
                                    ax.text(0.9, 0.95, note_to_add,
                                    transform=ax.transAxes, ha='right', va='top',
                                    fontsize=9, color='red', style='italic',
                                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 4})

                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plots_data[f"lightcurve_{band_name}_{plot_key}"] = buf.getvalue()
                plt.close(fig)

        except Exception as e:
            logging.error(f"Failed to create lightcurve for {obs_id} band {band_name}", exc_info=True)
            raise

    return plots_data

def create_pds_nicer(events, obs_id, segment_size=50.0, dt=0.001):
    
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