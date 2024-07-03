# i_pwr2.py

# imports
import os
import numpy as np
import soundfile as sf
import resampy
from aa_common import get_base, get_tmp_folder, interpolate_seg

# Function to adjust sample rate based on wavecycle length
def adjust_sample_rate_based_on_wavecycle_length(input_file_path, target_length, output_folder):
    data, original_sr = sf.read(input_file_path)
    target_sr = original_sr  # Default to original sample rate

    # Adjust the sample rate based on the target wavecycle length
    if 2731 < target_length < 5835:
        target_sr = 96000
    elif 5836 < target_length < 9600:
        target_sr = 48000

    # Resample the audio if the target sample rate differs from the original
    if target_sr != original_sr:
        data_resampled = resampy.resample(data, orig_sr=original_sr, sr_new=target_sr)
        new_filename = f"{os.path.splitext(os.path.basename(input_file_path))[0]}_{target_sr}.wav"
        output_file_path = os.path.join(output_folder, new_filename)
        sf.write(output_file_path, data_resampled, target_sr)
        # print(f"Resampled {os.path.basename(input_file_path)} to {target_sr}Hz, saved as {new_filename}.")
    else:
        output_file_path = os.path.join(output_folder, os.path.basename(input_file_path))
        sf.write(output_file_path, data, original_sr)
        # print(f"Copied {os.path.basename(input_file_path)} to {output_folder} without resampling.")

def run(single_cycles_192k32b, wavecycle_samples_target_192):
    base = get_base()
    tmp_folder = get_tmp_folder()

    # Define the output folder for 2048 sample segments
    output_folder_2048 = os.path.join(tmp_folder, '2048')
    os.makedirs(output_folder_2048, exist_ok=True)

    # Adjust sample rates and save to tmp folder
    for filename in os.listdir(single_cycles_192k32b):
        if filename.endswith('.wav'):
            input_file_path = os.path.join(single_cycles_192k32b, filename)
            adjust_sample_rate_based_on_wavecycle_length(input_file_path, wavecycle_samples_target_192, tmp_folder)

    # Transform adjusted segments to fixed length of 2048 samples
    for filename in os.listdir(tmp_folder):
        if filename.endswith('.wav') and ('_48000' in filename or '_96000' in filename or filename in os.listdir(single_cycles_192k32b)):
            input_file_path = os.path.join(tmp_folder, filename)
            data, original_sr = sf.read(input_file_path)
            interpolated_data = interpolate_seg(data, original_sr, 2048)
            output_file_path = os.path.join(output_folder_2048, filename)
            sf.write(output_file_path, interpolated_data, original_sr, subtype='FLOAT')
            # print(f"Transformed {filename} to 2048 samples and saved to {output_folder_2048}")

    # Delete temporary files from tmp folder after processing
    for filename in os.listdir(tmp_folder):
        file_path = os.path.join(tmp_folder, filename)
        # Check that the file is not in the '2048' folder and is a .wav file
        if filename.endswith('.wav') and file_path not in [os.path.join(output_folder_2048, f) for f in os.listdir(output_folder_2048)]:
            os.remove(file_path)
            # print(f"Deleted temporary file: {filename}")

    # Delete any other folders in tmp_folder except '2048'
    for folder in os.listdir(tmp_folder):
        folder_path = os.path.join(tmp_folder, folder)
        if os.path.isdir(folder_path) and folder != '2048':
            # First, delete all files in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # print(f"Deleted file: {file_path}")
            # Now delete the empty folder
            os.rmdir(folder_path)
            print(f"Deleted temporary folder: {folder_path}")

    print("i_pwr2 is running and completed")

if __name__ == "__main__":
    # Example call with dummy values (replace with actual values)
    single_cycles_192k32b = "/path/to/single_cycles_192k32b"
    wavecycle_samples_target_192 = 512  # Example value
    run(single_cycles_192k32b, wavecycle_samples_target_192)
