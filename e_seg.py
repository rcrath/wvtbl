
# e_seg.py
import os
import aa_common
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from aa_common import get_segment_sizes


def is_rising_zero_crossing(data, index):
    if index <= 0 or index >= len(data) - 1:
        return False
    if data[index - 1] < 0 and data[index] >= 0:
        return True
    return False

def is_full_wavecycle(segment, amplitude_tolerance_db=-60):
    if len(segment) < 3:
        return False

    first_sample_db = 20 * np.log10(max(abs(segment[0]), 1e-10))
    last_sample_db = 20 * np.log10(max(abs(segment[-1]), 1e-10))

    if first_sample_db > amplitude_tolerance_db or last_sample_db > amplitude_tolerance_db:
        return False

    zero_crossings = np.where(np.diff(np.signbit(segment)))[0]

    if len(zero_crossings) < 2:
        return False

    return True

def run(freq_est):
    # print("e_seg is running")

    # Initialize settings with defaults and update them
    settings = aa_common.initialize_settings()
    settings = aa_common.update_settings(settings)

    # Use the most recent settings
    settings = aa_common.update_settings(settings)

    # Use freq_est and other calculated values as needed
    plus_minus_tolerance = settings['percent_tolerance'] / 100.0
    wavecycle_samples_target_192 = round(192000 / freq_est)
    lower_bound = wavecycle_samples_target_192 * (1 - plus_minus_tolerance)
    upper_bound = wavecycle_samples_target_192 * (1 + plus_minus_tolerance)
    lower_bound_frequency = 192000 / upper_bound
    upper_bound_frequency = 192000 / lower_bound

    # Ensure tmp folder is created
    aa_common.ensure_tmp_folder()

    base = aa_common.get_base()
    tmp_folder = aa_common.get_tmp_folder()
    ext = ".wav"  # Assuming .wav extension
    base_prep_192k32b_path = os.path.join(tmp_folder, f"{base}_prep_192k32b.wav")

    # --- begin "segmentation" section ---
    segment_sizes = []  # Initialize the list to hold segment sizes
    prev_start_index = 0  # Start from the beginning
    amplitude_tolerance_db = -60  # Set amplitude tolerance in dB
    some_small_amplitude = 10 ** (amplitude_tolerance_db / 20)  # Convert to linear scale

    data, samplerate = sf.read(base_prep_192k32b_path)

    # print(f"Read file {base_prep_192k32b_path} with sample rate {samplerate} and {len(data)} samples")

    # Process the first segment explicitly if the start is near zero
    if abs(data[0]) < some_small_amplitude:
        for i in range(1, len(data)):
            if is_rising_zero_crossing(data, i):
                prev_start_index = i
                break  # Found the first real zero crossing, move to normal processing

        # Variables to hold the first two segments temporarily
    first_segment = None
    second_segment = None
    segment_index = 0

    # Iterate over the data to segment it
    for i in range(1, len(data)):
        if is_rising_zero_crossing(data, i):
            # Check if the current index is different from the previous start index
            if i != prev_start_index:
                wave_cycle = data[prev_start_index:i]
                prev_start_index = i
                # Store the size of this segment
                segment_sizes.append(len(wave_cycle))
                # Only proceed if wave_cycle is not empty
                if len(wave_cycle) > 0:
                    # Store the first two segments temporarily for special processing
                    if segment_index == 0:
                        first_segment = wave_cycle
                    elif segment_index == 1:
                        second_segment = wave_cycle

                    # Write segment to file if it's not the first or second segment
                    if segment_index > 1:
                        base_seg = f"{base}_seg_{segment_index:04d}{ext}"
                        tmp_base_seg_path = os.path.join(tmp_folder, base_seg)
                        sf.write(tmp_base_seg_path, wave_cycle, samplerate=192000, format='WAV', subtype='FLOAT')
                        # print(f"Segment {segment_index} written to {tmp_base_seg_path} with {len(wave_cycle)} samples")
                    segment_index += 1

    # Check if the first two segments contain full wave cycles
    if first_segment is not None and second_segment is not None:
        full_cycle_first = is_full_wavecycle(first_segment, amplitude_tolerance_db)
        full_cycle_second = is_full_wavecycle(second_segment, amplitude_tolerance_db)

        if not full_cycle_first or not full_cycle_second:
            # Combine the first two segments
            combined_segment = np.concatenate((first_segment, second_segment))

            # Write the combined segment to the '0001' file
            combined_path = os.path.join(tmp_folder, f"{base}_seg_0001{ext}")
            sf.write(combined_path, combined_segment, samplerate=192000, format='WAV', subtype='FLOAT')
            # print(f"Combined segment written to {combined_path} with {len(combined_segment)} samples")

            # Delete the '0000' file if it exists
            first_path = os.path.join(tmp_folder, f"{base}_seg_0000{ext}")
            if os.path.exists(first_path):
                os.remove(first_path)
                # print(f"Deleted {first_path}")
        else:
            # If both segments are full cycles, write them out as normal
            for i, segment in enumerate([first_segment, second_segment], start=0):
                segment_path = os.path.join(tmp_folder, f"{base}_seg_{i:04d}{ext}")
                sf.write(segment_path, segment, samplerate=192000, format='WAV', subtype='FLOAT')
                # print(f"Segment {i} written to {segment_path}")

    # Handle the last segment
    if prev_start_index < len(data):
        wave_cycle = data[prev_start_index:]
        # Check if the wave cycle is full before writing to file
        if is_full_wavecycle(wave_cycle, amplitude_tolerance_db) and len(wave_cycle) > 0:
            tmp_base_seg_path = os.path.join(tmp_folder, f"{base}_seg_{segment_index:04d}{ext}")
            sf.write(tmp_base_seg_path, wave_cycle, samplerate=192000, format='WAV', subtype='FLOAT')
            # print(f"Final segment {segment_index} written to {tmp_base_seg_path} with {len(wave_cycle)} samples")

if __name__ == "__main__":
    freq_est = 440  # Example frequency, replace with actual value from d_pitch
    run(freq_est)
