import os
import resampy
import numpy as np
import soundfile as sf
import math
from datetime import datetime
import matplotlib.pyplot as plt
from aa_common import get_base, get_tmp_folder, interpolate_seg
import ja_option3  # Import the ja_option3 module

selected_segment = None

# Functions
def concatenate_files(input_folder, output_file):
    all_frames = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            data, sr = sf.read(file_path)
            all_frames.append(data)
    
    if all_frames:
        wavetable_data = np.concatenate(all_frames, axis=0)
        sf.write(output_file, wavetable_data, sr, subtype='FLOAT')
        print(f"Saved concatenated wavetable to {output_file}")
        return output_file, sr
    else:
        raise ValueError("No valid .wav files found for concatenation.")

def apply_padding_if_needed(data, target_length, amplitude_tolerance_db):
    """Apply padding to the data if it's shorter than the target length."""
    current_length = len(data)
    if current_length < target_length:
        needed_samples = target_length - current_length
        padding = generate_drunken_walk(needed_samples, amplitude_tolerance_db)
        data_with_padding = np.concatenate([data, padding])
    else:
        data_with_padding = data
    return data_with_padding

def generate_drunken_walk(length, amplitude_db):
    """Generate a 'drunken walk' signal with a specified amplitude in dB."""
    # Convert amplitude from dB to a linear scale
    linear_amplitude = 10 ** (amplitude_db / 20.0)
    
    # Generate random steps, then perform a cumulative sum to simulate a 'walk'
    steps = np.random.uniform(low=-linear_amplitude, high=linear_amplitude, size=length)
    drunken_walk = np.cumsum(steps)
    
    # Ensure the signal stays within the desired amplitude range
    drunken_walk = np.clip(drunken_walk, -linear_amplitude, linear_amplitude)
    
    return drunken_walk

def split_and_save_wav_with_correct_padding(file_path, output_folder, base_name, wavetable_type, num_full_files):
    data, sr = sf.read(file_path) 
    segment_length = len(data)
    num_frames_per_file = 2048 * 256  # Serum-compatible wavetable size

    # Calculate the number of full wavetable files that can be created
    num_full_files = segment_length // num_frames_per_file

    # Process each full file
    for i in range(num_full_files):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        start_sample = i * num_frames_per_file
        end_sample = start_sample + num_frames_per_file
        segment = data[start_sample:end_sample]
        output_file_name = f"{base_name}_{wavetable_type}_{timestamp}.wav"
        output_file_path = os.path.join(output_folder, output_file_name)
        sf.write(output_file_path, segment, sr)  # Use `sr` to maintain original sample rate

    # Handle the last segment if there's a remainder, ensuring it also becomes 524288 samples long
    remainder = segment_length % num_frames_per_file
    if remainder > 0:
        padding_needed = num_frames_per_file - remainder
        if num_full_files > 0:
            padding_start = (num_full_files - 1) * num_frames_per_file + (num_frames_per_file - padding_needed)
            padding_samples = data[padding_start:padding_start + padding_needed]
            last_segment = np.concatenate([padding_samples, data[-remainder:]])
        else:
            last_segment = np.concatenate([data[:padding_needed], data[-remainder:]])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file_name = f"{base_name}_{wavetable_type}_{timestamp}.wav"
        output_file_path = os.path.join(output_folder, output_file_name)
        sf.write(output_file_path, last_segment, sr)  # Again, use `sr` for sample rate

def process_and_pad_wavetables(concat_file_path, output_folder, output_base_name):
    frame_length = 2048
    total_frames = 256
    total_samples_required = frame_length * total_frames

    # Read the concatenated file
    data, sr = sf.read(concat_file_path)

    # Check if the data length is less than required
    if len(data) < total_samples_required:
        # Calculate how many samples of silence are needed
        silence_padding_length = total_samples_required - len(data)
        # Create silence padding array
        silence_padding = np.zeros(silence_padding_length, dtype=data.dtype)
        # Append silence padding to the data
        data = np.concatenate((data, silence_padding))
    elif len(data) > total_samples_required:
        # If the data is longer than needed, truncate it
        data = data[:total_samples_required]

    # Ensure data is exactly the required length
    assert len(data) == total_samples_required, "Data length does not match the required wavetable length."

    # Save the processed wavetable
    output_file_path = os.path.join(output_folder, f"{output_base_name}_wavetable.wav")
    sf.write(output_file_path, data, sr)
    print(f"Wavetable saved to {output_file_path} with {len(data)} samples.")

def create_wavetable_from_concat(concat_file_path, output_wavetable_path):
    frame_length = 2048
    total_frames = 256
    total_samples_required = frame_length * total_frames

    # Load the concat file
    concat_data, sr = sf.read(concat_file_path)

    # General case: Keep halving the number of samples per wavecycle until it fits
    while len(concat_data) > total_samples_required:
        new_length = len(concat_data) // 2
        concat_data = resampy.resample(concat_data, sr, sr // 2, axis=-1)[:new_length]
    
    # Ensure the result is exactly 2048 * 256 samples long
    if len(concat_data) < total_samples_required:
        # Pad with silence if shorter
        padded_data = np.pad(concat_data, (0, total_samples_required - len(concat_data)), 'constant', constant_values=(0, 0))
    elif len(concat_data) > total_samples_required:
        # Truncate if longer (shouldn't happen with the above loop, but just in case)
        padded_data = concat_data[:total_samples_required]
    else:
        padded_data = concat_data

    # Save the padded data as the final wavetable
    sf.write(output_wavetable_path, padded_data, sr)
    print(f"Wavetable saved to {output_wavetable_path} with {len(padded_data)} samples.")

def run():
    global selected_segment
    base = get_base()
    tmp_folder = get_tmp_folder()
    
    output_folder_2048 = os.path.join(tmp_folder, '2048')
    concat_folder = os.path.join(tmp_folder, 'concat')
    os.makedirs(concat_folder, exist_ok=True)
    base_folder = os.path.join(os.getcwd(), base)
    os.makedirs(base_folder, exist_ok=True)
    
    output_file_2048 = f"{base}_2048_all.wav"
    output_path_2048 = os.path.join(concat_folder, output_file_2048)
    print(f"Output path for concatenated file: {output_path_2048}")
    
    output_path_2048, sr_2048 = concatenate_files(output_folder_2048, output_path_2048)
    print(f"Concatenated file created at: {output_path_2048} with sample rate: {sr_2048}")
    
    final_wavetable_path = os.path.join(base_folder, f"{base}_2048_wvtbl_all.wav")
    
    option = input("Choose wavetable creation method:\n1. Squished to 256*2048 samples\n2. Arbitrarily cut into chunks\n3. Select and save\n")
    
    if option == '1':
        create_wavetable_from_concat(output_path_2048, final_wavetable_path)
        print(f"Final wavetable saved to: {final_wavetable_path}")
    elif option == '2':
        data_2048, sr_2048 = sf.read(output_path_2048, dtype='float32')
        data_2048_padded = apply_padding_if_needed(data_2048, 524288, -60)
        sf.write(output_path_2048, data_2048_padded, sr_2048)
        
        num_full_files_2048 = math.ceil(len(data_2048_padded) / 524288)
        split_and_save_wav_with_correct_padding(output_path_2048, base_folder, base, "2048", num_full_files_2048)
        print(f"Wavetable split and saved in: {base_folder}")
    elif option == '3':
        ja_option3.plot_wav_file_interactive(output_path_2048)
        if ja_option3.selected_segment is not None:
            ja_option3.save_selection(ja_option3.selected_segment, sr_2048)
        else:
            print("No selection was made.")
    else:
        print("Invalid option")

    print("j_wvtblr is running and completed")

if __name__ == "__main__":
    run()

