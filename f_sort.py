# f_sort.py
import os
import aa_common
import numpy as np
import soundfile as sf
from aa_common import get_segment_sizes, print_segment_info

def calculate_tolerance_bounds(wavecycle_samples_target_192, tolerance_percent):
    plus_minus_tolerance = tolerance_percent / 100.0
    lower_bound = 1 - plus_minus_tolerance
    upper_bound = 1 + plus_minus_tolerance
    lower_bound_samples = wavecycle_samples_target_192 * lower_bound
    upper_bound_samples = wavecycle_samples_target_192 * upper_bound
    return lower_bound_samples, upper_bound_samples

def mark_deviant_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, tmp_folder, ext):
    outside_tolerance_files = []
    for segment_file, segment_size in segment_sizes:
        if segment_size < lower_bound_samples or segment_size > upper_bound_samples:
            if '_dev' not in segment_file:
                dev_name = f"{os.path.splitext(segment_file)[0]}_dev{ext}"
            else:
                dev_name = segment_file
            dev_path = os.path.join(tmp_folder, dev_name)
            file_path = os.path.join(tmp_folder, segment_file)
            os.rename(file_path, dev_path)
            outside_tolerance_files.append(dev_name)
    return outside_tolerance_files

def mark_attack_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, tmp_folder, ext):
    within_tolerance_count = 0
    attack_segments = []

    for segment_file, segment_size in segment_sizes:
        if lower_bound_samples <= segment_size <= upper_bound_samples:
            within_tolerance_count += 1
        else:
            within_tolerance_count = 0
        
        if within_tolerance_count < 3:
            attack_segments.append(segment_file)
        else:
            break

    # Rename attack segments
    for segment_file in attack_segments:
        if '_atk' not in segment_file:
            atk_name = f"{os.path.splitext(segment_file)[0]}_atk{ext}"
        else:
            atk_name = segment_file

        atk_path = os.path.join(tmp_folder, atk_name)
        file_path = os.path.join(tmp_folder, segment_file)

        if not os.path.exists(file_path):
            # Handle the case where the file is a deviant segment
            file_path = os.path.join(tmp_folder, f"{os.path.splitext(segment_file)[0]}_dev{ext}")
        
        if os.path.exists(file_path):
            os.rename(file_path, atk_path)
        else:
            print(f"Error: Could not find file {file_path} to rename to {atk_name}")

    return attack_segments

def run(freq_est, settings, discarding_atk=False, discarding_dev=False, discarding_good=False, first_iteration=True):
    # print("f_sort is running")
    
    base = aa_common.get_base()
    tmp_folder = aa_common.get_tmp_folder()
    ext = ".wav"

    # Calculate tolerance bounds in samples
    wavecycle_samples_target_192 = round(192000 / freq_est)
    lower_bound_samples, upper_bound_samples = calculate_tolerance_bounds(wavecycle_samples_target_192, settings['percent_tolerance'])
    # Get sizes of the segments
    segment_sizes = get_segment_sizes(base, tmp_folder, ext)
    total_segments = len(segment_sizes)

    # Mark deviant segments
    outside_tolerance_files = mark_deviant_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, tmp_folder, ext)
    total_deviant_segments = len(outside_tolerance_files)
    total_normal_segments = total_segments - total_deviant_segments

    # print (f"DEBUG:{segment_sizes} smpls")
    # print(f"{total_segments} ttl segs")
    # print(f"{wavecycle_samples_target_192} samples per cycle target, ranging from {lower_bound_samples} to {upper_bound_samples}")
    # print(f"{outside_tolerance_files} dev, {total_deviant_segments} ttl dev, {total_normal_segments}ttl normal")

    # Mark attack segments
    attack_segments = mark_attack_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, tmp_folder, ext)
    total_attack_segments = len(attack_segments)

    # Print segment information if this is the first iteration
    if first_iteration:
        print_segment_info(
            total_segments,
            total_deviant_segments,
            total_normal_segments,
            total_attack_segments,
            freq_est,
            wavecycle_samples_target_192,
            settings,
            lower_bound_samples,
            upper_bound_samples,
            discarding_atk,
            discarding_dev,
            discarding_good
        )

    return total_segments, total_deviant_segments, total_normal_segments, total_attack_segments


if __name__ == "__main__":
    # Initialize and update settings before accessing any values
    settings = aa_common.initialize_settings()

    # Example values for testing; replace with actual values from d_pitch
    freq_est = 440  # Example frequency
    
    # Ensure settings dictionary contains these keys if used
    settings['freq_est'] = freq_est
    
    # Run f_sort with the updated settings
    total_segments, total_deviant_segments, total_normal_segments, total_attack_segments = run(freq_est, settings)
