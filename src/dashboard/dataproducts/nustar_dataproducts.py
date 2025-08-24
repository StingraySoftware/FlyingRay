import logging
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from stingray import EventList
from stingray.deadtime.fad import FAD, get_periodograms_from_FAD_results, calculate_FAD_correction
from stingray.gti import get_gti_lengths
from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum


logger = logging.getLogger(__name__)

def create_energy_band_lightcurves_nustar(events_a, events_b, obs_id, dt=10):
    large_gap_threshold=3600
    plots_data = {}
    energy_bands = {'3-79_keV': (3, 79)}

    for band_name, (e_min, e_max) in energy_bands.items():
        try:
            events_a_filt = (events_a.energy >= e_min) & (events_a.energy <= e_max)
            filtered_times_a = events_a.time[events_a_filt]

            if len(filtered_times_a) == 0:
                logging.warning(f"No FPMA events in band {band_name} for {obs_id}.")
                continue
                
            lc_det_a = Lightcurve.make_lightcurve(filtered_times_a, dt=dt, gti=events_a.gti)
            lc_det_a.apply_gtis()
            lc_det_a.gti = lc_det_a.gti - lc_det_a.time[0]
            lc_det_a.time = lc_det_a.time - lc_det_a.time[0]

            if lc_det_a.time is None or len(lc_det_a.time) == 0:
                logging.warning(f"No data left after applying GTIs for {obs_id} band {band_name}. Skipping.")
                continue
       
            events_b_filt = (events_b.energy >= e_min) & (events_b.energy <= e_max)
            filtered_times_b = events_b.time[events_b_filt]
            
            if len(filtered_times_b) == 0:
                logging.warning(f"No FPMB events in band {band_name} for {obs_id}.")
                continue
                
            lc_det_b = Lightcurve.make_lightcurve(filtered_times_b, dt=dt, gti=events_b.gti)
            lc_det_b.apply_gtis()
            lc_det_b.gti = lc_det_b.gti - lc_det_b.time[0]
            lc_det_b.time = lc_det_b.time - lc_det_b.time[0]

            if lc_det_b.time is None or len(lc_det_b.time) == 0:
                logging.warning(f"No data left after applying GTIs for {obs_id} band {band_name}. Skipping.")
                continue
                
            note_to_add = None
            combined_counts = np.concatenate((lc_det_a.countrate, lc_det_b.countrate))
            valid_counts = combined_counts[np.isfinite(combined_counts) & (combined_counts > 0)]
            if len(valid_counts) > 1:
                min_intensity = np.min(valid_counts)
                max_intensity = np.max(valid_counts)
                if min_intensity > 0:
                    ratio = max_intensity / min_intensity
                    if ratio > 50:
                        variability_note = "NOTE: High variability detected (Max/Min > 50). Consider custom processing flags."
                        

            all_gtis = sorted([tuple(g) for g in lc_det_a.gti], key=lambda x: x[0]) if lc_det_a.gti is not None else [] 
            plots_to_generate = {}
            
            if len(all_gtis) <= 2 or len(all_gtis) > 10:
                 plots_to_generate['full_continuous_lc'] = {'title': f"Light Curve {band_name}: {obs_id} (dt={dt}s)", 'plot_type': 'simple_continuous'}
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
                    ax.plot(lc_det_a.time, lc_det_a.countrate, color='dodgerblue', marker='o', markersize=2, linestyle='-', linewidth=0.1) 
                    ax.plot(lc_det_b.time, lc_det_b.countrate, color='orangered', marker='x', markersize=2, linestyle='-', linewidth=0.1)

                    if lc_det_a.gti is not None and len(lc_det_a.gti) > 1:
                        for i in range(len(lc_det_a.gti) - 1):
                            ax.axvspan(lc_det_a.gti[i, 1], lc_det_a.gti[i + 1, 0], alpha=0.3, color='red', zorder=0)
                    if len(lc_det_a.time) > 1:
                        ax.set_xlim(lc_det_a.time[0], lc_det_a.time[-1])
   
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
                    #axes = np.atleast_1d(axes)
                    axes = np.array(axes).flatten()

                    for i, segment_gtis in enumerate(populated_segments):
                        ax = axes[i]
                        start_time, end_time = segment_gtis[0][0], segment_gtis[-1][1]
                        
                        # Plot FPMA segment
                        mask_a = (lc_det_a.time >= start_time) & (lc_deta-a.time <= end_time)
                        if np.any(mask_a):
                            ax.plot(lc_det_a.time[mask_a], lc_det_a.countrate[mask_a], color='dodgerblue', marker='o', markersize=2, linestyle='-', linewidth=0.1)

                        # Plot FPMB segment
                        mask_b = (lc_det_b.time >= start_time) & (lc_det_b.time <= end_time)
                        if np.any(mask_b):
                            ax.plot(lc_det_b.time[mask_b], lc_det_b.countrate[mask_b], color='orangered', marker='x', markersize=2, linestyle='-', linewidth=0.1)
                        

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
                        last_digits_start = int(start_time) % divisor if int(start_time) > 0 else 0
                        last_digits_end = int(end_time)                       
                        time_range_label = f"{last_digits_start} - {last_digits_end}"
                      
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
                    if note_to_add:
                        ax.text(0.9, 0.95, note_to_add,
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=9, color='red', style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 4})

                    
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
            raise

    return plots_data

def create_pds_nustar(events_a, events_b, obs_id, segment_size=100.0, dt=0.001):
    """
    Creates a NuSTAR PDS.

    First, it attempts to calculate the dead-time corrected PDS using the FAD
    method. If this fails (e.g., due to data gaps or low counts), it logs a
    warning and automatically falls back to calculating a standard, uncorrected
    PDS.

    """
    try:
        lc_a = events_a.to_lc(dt)
        lc_b = events_b.to_lc(dt)

        if lc_a.counts.sum() == 0 or lc_b.counts.sum() == 0:
            raise ValueError("Light curve is empty, cannot perform FAD.")
        fad_results_table = calculate_FAD_correction(lc1=lc_a, lc2=lc_b, segment_size=segment_size, norm="frac")
        pds_a = get_periodograms_from_FAD_results(fad_results_table, kind='pds1')
        pds_a_reb = pds_a.rebin_log(0.03)
        noise_powers_a = pds_a.power[pds_a.freq > 100]
        noise_a = np.mean(noise_powers_a) if len(noise_powers_a) > 0 else 0
        y_vals_a = (pds_a_reb.power - noise_a) * pds_a_reb.freq
        #y_err_a = (pds_a_reb.power / np.sqrt(pds_a_reb.m)) * pds_a_reb.freq

        # --- PDS for Detector B (FPMB) ---
        pds_b = get_periodograms_from_FAD_results(fad_results_table, kind='pds2')
        pds_b_reb = pds_b.rebin_log(0.03)
        noise_powers_b = pds_b.power[pds_b.freq > 100]
        noise_b = np.mean(noise_powers_b) if len(noise_powers_b) > 0 else 0
        y_vals_b = (pds_b_reb.power - noise_b) * pds_b_reb.freq
       # y_err_b = (pds_b_reb.power / np.sqrt(pds_b_reb.m)) * pds_b_reb.freq

        valid_mask_a = (y_vals_a > 0) & np.isfinite(y_vals_a)
        valid_mask_b = (y_vals_b > 0) & np.isfinite(y_vals_b)
        
        if not np.any(valid_mask_a) and not np.any(valid_mask_b):
         raise ValueError("FAD resulted in no positive power values.")

        fig, ax = plt.subplots()
        plot_title = f"NuSTAR FAD-Corrected PDS: {obs_id}"
        ax.plot(pds_a_reb.freq[valid_mask_a], y_vals_a[valid_mask_a], drawstyle="steps-mid", color="dodgerblue")
        
        ax.plot(pds_b_reb.freq[valid_mask_b], y_vals_b[valid_mask_b], drawstyle="steps-mid", color="orangered",alpha=0.7)

             
    except Exception as fad_error:
        logger.warning(f"FAD correction failed for {obs_id}: {fad_error}. Falling back to standard PDS.")
        try:
        # --- PDS for Detector A (FPMA) ---
         pds_a = AveragedPowerspectrum.from_events(
            events_a, segment_size=segment_size, dt=dt, norm="frac", use_common_mean=True
         )
         pds_a_reb = pds_a.rebin_log(0.03)
         noise_powers_a = pds_a.power[pds_a.freq > 100]
         noise_a = np.mean(noise_powers_a) if len(noise_powers_a) > 0 else 0
         y_vals_a = (pds_a_reb.power - noise_a) * pds_a_reb.freq

        # --- PDS for Detector B (FPMB) ---
         pds_b = AveragedPowerspectrum.from_events(
            events_b, segment_size=segment_size, dt=dt, norm="frac", use_common_mean=True
         )
         pds_b_reb = pds_b.rebin_log(0.03)
         noise_powers_b = pds_b.power[pds_b.freq > 100]
         noise_b = np.mean(noise_powers_b) if len(noise_powers_b) > 0 else 0
         y_vals_b = (pds_b_reb.power - noise_b) * pds_b_reb.freq
         valid_mask_a = (y_vals_a > 0) & np.isfinite(y_vals_a)
         valid_mask_b = (y_vals_b > 0) & np.isfinite(y_vals_b)

         if not np.any(valid_mask_a) and not np.any(valid_mask_b):
           raise ValueError("FAD resulted in no positive power values.")

         fig, ax = plt.subplots()
         plot_title = f"NuSTAR Standard PDS: {obs_id} (FAD Failed)"
        
         ax.plot(pds_a_reb.freq[valid_mask_a], y_vals_a[valid_mask_a], drawstyle="steps-mid", color="dodgerblue")
        
         ax.plot(pds_b_reb.freq[valid_mask_b], y_vals_b[valid_mask_b], drawstyle="steps-mid", color="orangered",alpha=0.7)

        except Exception:
            logger.error(f"FATAL: Both FAD and Standard PDS failed for {obs_id}", exc_info=True)
            return None

    if 'fig' in locals() and 'ax' in locals():
     ax.loglog()
     ax.set_xlabel("Frequency (Hz)")
     ax.set_ylabel(r"Power $\times$ Frequency [$(\mathrm{rms/mean})^2$]")
     ax.set_title(plot_title)

     x_lim_bottom = 1. / segment_size
     x_lim_top = 1. / (2. * dt)
     ax.set_xlim(left=x_lim_bottom, right=x_lim_top)

     all_y_vals = np.concatenate((y_vals_a[valid_mask_a], y_vals_b[valid_mask_b]))
     if len(all_y_vals) > 0:
            y_min = np.min(all_y_vals)
            y_max = np.max(all_y_vals)
            ax.set_ylim(bottom=y_min * 0.5, top=y_max * 2.0)


     buf = BytesIO()
     fig.savefig(buf, format='png', dpi=100)
     buf.seek(0)
     plot_data = buf.getvalue()
     plt.close(fig)

    return plot_data
