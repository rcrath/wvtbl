import os
import b_menu
import c_upsample
import d_pitch
import e_seg
import f_sort
import g_choose
import h_interpolate
import i_pwr2
import j_wvtblr
import k_clean  # Import the cleanup module
import aa_common

def main():
    # Initialize settings
    settings = aa_common.initialize_settings()

    # Run the menu to select the source file
    b_menu.run()

    # Run the upsampling
    c_upsample.run()

    # Run pitch detection and get frequency estimation
    freq_est = d_pitch.run()

    tmp_folder = aa_common.get_tmp_folder()
    prep_file_name = f"{aa_common.get_base()}_prep_192k32b.wav"

    # Initialize segment variables
    total_segments = 0
    total_deviant_segments = 0
    total_normal_segments = 0
    total_attack_segments = 0
    discarding_atk = False
    discarding_dev = False
    discarding_good = False

    first_iteration = True
    single_cycles_192k32b = os.path.join(tmp_folder, '192k32b_singles')
    wavecycle_samples_target_192 = round(192000 / freq_est)
    os.makedirs(single_cycles_192k32b, exist_ok=True)

    while True:
        # Run segmentation with the obtained frequency estimation
        e_seg.run(freq_est)

        # Label segments
        total_segments, total_deviant_segments, total_normal_segments, total_attack_segments = f_sort.run(freq_est, settings)

        # Choose settings and update freq_est if changed
        freq_est, settings, total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, discarding_atk, discarding_dev, discarding_good, first_iteration = g_choose.run(
            freq_est, settings, total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, first_iteration)

        # Check if the settings are accepted
        if 'accept_current_settings' in settings and settings['accept_current_settings']:
            break
        else:
            # Clean up tmp folder except for the prep file
            aa_common.cleanup_tmp_folder(tmp_folder, prep_file_name)
            
            # Update settings and re-run segmentation
            settings = aa_common.update_settings(settings)

    # Proceed to h_interpolate
    h_interpolate.run(total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, freq_est, settings, discarding_atk, discarding_dev, discarding_good)

    # Ensure the directory exists
    single_cycles_pwr2_any = os.path.join(tmp_folder, '192k_pwr2_any')
    os.makedirs(single_cycles_pwr2_any, exist_ok=True)

    # Proceed to i_pwr2
    i_pwr2.run(single_cycles_192k32b, wavecycle_samples_target_192)

    # Ensure the concat folder exists
    concat_folder = os.path.join(tmp_folder, 'concat')
    os.makedirs(concat_folder, exist_ok=True)

    # Proceed to j_wvtblr
    j_wvtblr.run()

    # Cleanup temporary files
    k_clean.run()

    print("\n\nSUCCESS\n\n")

if __name__ == "__main__":
    main()
