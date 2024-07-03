# g_choose.py
import os
import aa_common

def delete_segments(base, tmp_folder, ext, settings):
    segment_files = [f for f in os.listdir(tmp_folder) if f.startswith(f"{base}_seg") and f.endswith(ext)]

    # First, delete attack segments if requested
    if settings['discard_atk_choice'] == 'Y':
        for segment_file in segment_files:
            if '_atk' in segment_file:
                os.remove(os.path.join(tmp_folder, segment_file))
    
    # Then, delete deviant segments if requested
    if settings['discard_dev_choice'] == 'Y':
        for segment_file in segment_files:
            if '_dev' in segment_file:
                os.remove(os.path.join(tmp_folder, segment_file))
    
    # Finally, delete good segments if requested
    if settings['discard_good_choice'] == 'Y':
        for segment_file in segment_files:
            if '_seg' in segment_file and not any(suffix in segment_file for suffix in ['_atk', '_dev']):
                os.remove(os.path.join(tmp_folder, segment_file))

def run(freq_est, settings, total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, first_iteration):
    # print("g_choose is running")
    
    # Update settings with user input
    settings = aa_common.prompt_update_settings(freq_est, settings)

    # Use the updated freq_est from settings if it was changed
    if 'freq_note_input' in settings and settings['freq_note_input'] != 'enter':
        try:
            freq_est = float(settings['freq_note_input'])
        except ValueError:
            freq_est = aa_common.note_to_frequency(settings['freq_note_input'])
    
    # Perform cleanup based on user choices
    base = aa_common.get_base()
    tmp_folder = aa_common.get_tmp_folder()
    ext = ".wav"

    delete_segments(base, tmp_folder, ext, settings)

    # Recalculate segment information after deletion
    total_segments, total_deviant_segments, total_normal_segments, total_attack_segments = aa_common.recount_segments(base, tmp_folder, ext)

    discarding_atk = settings['discard_atk_choice'] == 'Y'
    discarding_dev = settings['discard_dev_choice'] == 'Y'
    discarding_good = settings['discard_good_choice'] == 'Y'

    return freq_est, settings, total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, discarding_atk, discarding_dev, discarding_good, False

if __name__ == "__main__":
    freq_est = 440  # Example frequency, replace with actual value from d_pitch
    settings = aa_common.initialize_settings()
    settings = aa_common.update_settings(settings)
    total_segments, total_deviant_segments, total_normal_segments, total_attack_segments = 0, 0, 0, 0
    first_iteration = True
    run(freq_est, settings, total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, first_iteration)
