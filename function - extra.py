functions

def amplitude_to_db(amplitude):
    # Prevent log of zero or negative values by setting a minimum amplitude level (e.g., 1e-10)
    amplitude[amplitude == 0] = 1e-10
    return 20 * np.log10(abs(amplitude))


# Function to check if a file is effectively silent (zero amplitude throughout)
def is_file_silent(start_file):
    data, _ = sf.read(start_file)  # Read the file
    return np.all(data == 0)  # Check if all values in data are 0

def prompt_for_start_frame(highest_frame):
    while True:
        start_frame_input = input(f"Enter the starting frame (1 [default] to {highest_frame}): ") or '1'
        try:
            start_frame = int(start_frame_input)
            if 1 <= start_frame <= highest_frame:
                return start_frame
            else:
                print(f"Please enter a number within the range 1 to {highest_frame}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# read the audio for use in next, test_crepe
def run_test_crepe(base_prep_192k32b_path):
    sr, audio = wavfile.read(base_prep_192k32b_path)
    time, frequency, confidence = test_crepe(audio, sr)
    return time, frequency, confidence

def combine_and_save_frames(start_frame, frame_count):
    combined_frames = []
    ending_frame = start_frame + frame_count - 1

    for filename in sorted(os.listdir(pwr2_192_2048))[start_frame - 1:ending_frame]:
        if filename.endswith(ext):
            file_2048_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_2048_path)
            combined_frames.append(waveform)

    combined_frame = np.concatenate(combined_frames, axis=0)
    combined_2048x256_file_name = f"{base}_2048x256_start{start_frame:04d}_{frame_count}_frame{ext}"
    combined_2048x256_frame_out_path = os.path.join(concat_folder, combined_2048x256_file_name)
    sf.write(combined_2048x256_frame_out_path, combined_frame, sr, subtype='FLOAT')
    print(f"Combined {frame_count} frames starting from frame {start_frame} saved as: {combined_2048x256_frame_out_path}")

def perform_backfill_and_invert():
    backward_fill_count = frames_to_combine_wt - total_files_in_single_cycles_192k32b
    combined_frames = []

    for filename in sorted(os.listdir(pwr2_192_2048))[:total_files_in_single_cycles_192k32b]:
        if filename.endswith(ext):
            file_2048_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_2048_path)
            combined_frames.append(waveform)
    
    for filename in sorted(os.listdir(pwr2_192_2048))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_2048_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_2048_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)
    
    combined_frame = np.concatenate(combined_frames, axis=0)
    backfill_file_name = f"{base}_combined_backfilled_{frames_to_combine_wt}_frames{ext}"
    backfill_file_path = os.path.join(pwr2_192_2048, backfill_file_name)
    sf.write(backfill_file_path, combined_frame, sr, subtype='FLOAT')
    print(f"Backfilled combined file created at {backfill_file_path}")

def calculate_pitch_shift_factor(mode_frequency, target_pitch, sr):
    return 12 * np.log2(target_pitch / mode_frequency)

# Function to find the closest multiple of 93.75 Hz to the original pitch
def closest_multiple_of_9375(original_pitch):
    multiple = round(original_pitch / 93.75)
    return multiple * 93.75

def cents_difference(freq1, freq2):
    # Calculate the difference in cents between two frequencies
    return 1200 * log2(freq2 / freq1)

#other def better
def get_manual_frequency_input(lowest_freq, highest_freq):
    while True:
        freq_note_input = input(f"Enter the frequency in Hz (between {lowest_freq}Hz and {highest_freq}Hz), or press <enter> to proceed without setting it.\nHz: ").strip()

        if not freq_note_input:
            return None  # User chose to skip manual input

        if freq_note_input.replace('.', '', 1).isdigit():
            freq_est = float(freq_note_input)
            if lowest_freq <= freq_est <= highest_freq:
                return freq_est  # Valid frequency; exit the loop
            else:
                print(f"The frequency {freq_est}Hz is out of bounds. Valid Frequencies are {lowest_freq} to {highest_freq}")
        else:
            print("Invalid input. Please enter a valid frequency in Hz.")

def calculate_pitch_at_sample_rate(original_pitch, original_sr, new_sr):
    return original_pitch * (new_sr / original_sr)

def find_nearest_harmonic(freq_est, mode_frequency):
    """
    Find the nearest multiple or submultiple of mode_frequency to freq_est.
    """
    # Calculate the ratio of freq_est to mode_frequency
    ratio = freq_est / mode_frequency
    
    # Find the nearest whole number to the ratio
    nearest_whole_number_ratio = round(ratio)
    
    # Calculate the adjusted frequency based on this ratio
    adjusted_freq = mode_frequency * nearest_whole_number_ratio
    
    return adjusted_freq, nearest_whole_number_ratio

def resample_to_power_of_two(single_cycles_192k32b, wavecycle_samples_target_192):
    nearest_192_higher_pwr2 = int(2**np.ceil(np.log2(wavecycle_samples_target_192)))

    # Define the single cycle folder named '192' to save the interpolated segments in
    base = get_base()
    tmp_folder = get_tmp_folder()
    single_cycles_pwr2_any = os.path.join(tmp_folder, f'192k_pwr2_{nearest_192_higher_pwr2}')

    # Create the single_cycles_pwr2_any folder if it doesn't exist
    if not os.path.exists(single_cycles_pwr2_any):
        os.makedirs(single_cycles_pwr2_any, exist_ok=True)

    print(f"Source 192 waveforms folder set to: {single_cycles_192k32b}")
    print(f"Resampled waveforms folder set to: {single_cycles_pwr2_any}")

    # Define the ratio of the nearest higher power of 2 to wavecycle_samples_target_192
    pwr_of_2_192_ratio = nearest_192_higher_pwr2 / wavecycle_samples_target_192

    # Calculate the target length for 192K files as the power of 2 target
    pwr_of_2_192_target = int(round(wavecycle_samples_target_192 * pwr_of_2_192_ratio))

    for filename in os.listdir(single_cycles_192k32b):
        if filename.endswith('.wav'):  # Check if the file is a .wav file
            # Construct the full path for the input file using single_cycles_192k32b
            input_file_path = os.path.join(single_cycles_192k32b, filename)

            # Read the waveform and its sample rate
            data, original_sr = sf.read(input_file_path)

            # Use interpolate_seg to resample the segment to the new target length
            interpolated_data = interpolate_seg(data, original_sr, pwr_of_2_192_target)

            # Construct the full path for the output file using single_cycles_pwr2_any
            pwr2_192_file_path = os.path.join(single_cycles_pwr2_any, filename)

            # Save the interpolated segment to the single_cycles_pwr2_any
            sf.write(pwr2_192_file_path, interpolated_data, original_sr, subtype='FLOAT')
            # print(f"Processed and saved: {filename} to {single_cycles_pwr2_any}")

    return single_cycles_pwr2_any



