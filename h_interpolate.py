# h_interpolate.py

# imports
import os
import numpy as np
import soundfile as sf
import resampy
from aa_common import get_segment_sizes, print_segment_info, get_base, get_tmp_folder, interpolate_seg

# Function to interpolate a segment using resampy

def run(total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, freq_est, settings, discarding_atk, discarding_dev, discarding_good):
    print("h_interpolate is running")
    # define variables
    base = get_base()
    tmp_folder = get_tmp_folder()
    ext = ".wav"
    # Get sizes of the segments
    segment_sizes = get_segment_sizes(base, tmp_folder, ext)
    total_segments = len(segment_sizes)
    wavecycle_samples_target_192 = 192000 // freq_est
    lower_bound_samples, upper_bound_samples = 0, 0  # Replace with actual calculation if available

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

    # block 7
    # ---INTERPOLATION ---
    print("\nInterpolating...")

    # Create a new subfolder '192k32b_singles' within 'tmp_folder'
    singles_folder = os.path.join(tmp_folder, '192k32b_singles')
    os.makedirs(singles_folder, exist_ok=True)

    # Iterate through all files in the tmp folder
    for file in os.listdir(tmp_folder):
        # Check if the file name follows the expected pattern
        if file.startswith(f"{base}_seg_") and file.endswith(".wav"):
            # Extract the segment index and any suffix after the index
            prefix_length = len(f"{base}_seg_")
            segment_idx_str = file[prefix_length:prefix_length+4]
            suffix = file[prefix_length+4:-4]  # Exclude the '.wav' at the end

            # Ensure the extracted segment index is all digits
            if segment_idx_str.isdigit():
                seg_file_path = os.path.join(tmp_folder, file)

                # Read the original segment and get its sample rate and subtype
                data, samplerate = sf.read(seg_file_path)
                info = sf.info(seg_file_path)

                # Determine the correct subtype for writing based on the subtype of the original file
                write_subtype = 'FLOAT'  # Default to FLOAT for compatibility
                if info.subtype in ['PCM_16', 'PCM_24', 'PCM_32']:
                    write_subtype = info.subtype

                segment_idx = int(segment_idx_str)
                if segment_idx >= len(segment_sizes):
                    print(f"Segment index out of range for file {file}")
                    continue  # Skip this file and proceed to the next one

                # Use the target length for interpolation
                target_length = wavecycle_samples_target_192

                # Apply interpolation to adjust the segment length to the target length
                interpolated_segment = interpolate_seg(data, samplerate, target_length)

                # Construct the new file name using the captured parts
                single_cycles_192k32b_name = f"{base}_seg_{segment_idx_str}{suffix}.wav"
                single_cycles_192k32b_path = os.path.join(singles_folder, single_cycles_192k32b_name)
        
                # Write the interpolated segment to the '192k32b_singles' folder
                sf.write(single_cycles_192k32b_path, interpolated_segment, samplerate, subtype=write_subtype)

    total_segments_count = 0
    attack_segments_count = 0
    deviant_segments_count = 0
    good_segments_count = 0

    # Iterate through all files in the 'singles_folder' to count
    for file in os.listdir(singles_folder):
        if file.endswith(".wav"):
            total_segments_count += 1
            if '_atk' in file:
                attack_segments_count += 1
            elif '_dev' in file:
                deviant_segments_count += 1
            else:
                good_segments_count += 1

    print("\nConstructing wavetables ...")

if __name__ == "__main__":
    # Example call with dummy values (replace with actual values)
    run(0, 0, 0, 0, 440, {}, False, False, False)
