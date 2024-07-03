

# c_upsample.py

import os
import aa_common
import librosa
import resampy
import numpy as np
import soundfile as sf

def load_audio(file_path):
    return librosa.load(file_path, sr=None)

def interpolate_best(waveform, original_sr, target_sr):
    """
    Resample a waveform to a new sample rate using high-quality resampling (suitable for both upsampling and downsampling).
    
    Parameters:
    - waveform: np.ndarray, the input waveform (audio signal).
    - original_sr: int, the original sample rate (e.g., 48000 for 48kHz).
    - target_sr: int, the target sample rate (e.g., 192000 for 192kHz or any other rate for downsampling).
    
    Returns:
    - np.ndarray, the resampled waveform.
    """
    # print(f"Starting resampling from {original_sr} Hz to {target_sr} Hz")
    resampled_waveform = resampy.resample(waveform, original_sr, target_sr)
    # print(f"Resampling complete. Resampled waveform length: {len(resampled_waveform)}")
    return resampled_waveform

def apply_fades(data, fade_samples):
    if len(data) > 2 * fade_samples:
        fade_in_window = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out_window = np.linspace(1, 0, fade_samples, dtype=np.float32)
        data[:fade_samples] *= fade_in_window
        data[-fade_samples:] *= fade_out_window
    else:
        print("Audio data too short for the specified fade length.")
    return data

def save_audio(file_path, data, sample_rate):
    """
    Save the audio data to a specified file path with the given sample rate.
    
    Parameters:
    - file_path: str, the path to save the audio file.
    - data: np.ndarray, the audio data to save.
    - sample_rate: int, the sample rate of the audio data.
    """
    sf.write(file_path, data, sample_rate, subtype='FLOAT')

def run():
    # print("Running c_upsample")
    
    start_file = aa_common.get_start_file()
    tmp_folder = aa_common.get_tmp_folder()
    base = aa_common.get_base()

    # Print paths for debugging
    # print(f"start_file: {start_file}")
    # print(f"tmp_folder: {tmp_folder}")
    # print(f"base: {base}")

    # Load the waveform and sample rate from the input file
    start_file_data, sample_rate = load_audio(start_file)
    # print(f"Loaded audio file '{start_file}' with sample rate {sample_rate} Hz")

    # Calculate the duration of the input waveform in seconds
    duration_seconds = librosa.get_duration(y=start_file_data, sr=sample_rate)
    # print(f"Audio duration: {duration_seconds} seconds")

    # Calculate the number of samples needed to interpolate to 192k while keeping the same duration
    target_samples_192k = round(192000 * duration_seconds)
    # print(f"Target number of samples at 192kHz: {target_samples_192k}")

    # Resample the input waveform to 192k samples using the best interpolation method
    # print("Resampling the input waveform to 192kHz...")
    interpolated_input_192k32b = interpolate_best(start_file_data, sample_rate, 192000)
    # print("Resampling completed.")

    # Save the interpolated input as a temporary wave file
    base_prep_192k32b = f"{base}_prep_192k32b.wav"
    base_prep_192k32b_path = os.path.join(tmp_folder, base_prep_192k32b)
    # print(f"Saving file to: {base_prep_192k32b_path}")
    save_audio(base_prep_192k32b_path, interpolated_input_192k32b, 192000)

    # Confirm the file creation
    # if os.path.exists(base_prep_192k32b_path):
    #     print(f"File created: {base_prep_192k32b_path}")
    # else:
    #     print(f"Failed to create file: {base_prep_192k32b_path}")
    #     return

    # Load the upsampled file
    base_prep_192k32b_data, _ = load_audio(base_prep_192k32b_path)

    # Define the number of samples for fade in and fade out
    fade_samples = 2048  # Adjust this as needed

    # Apply fades
    base_prep_192k32b_data = apply_fades(base_prep_192k32b_data, fade_samples)

    # Save the audio data with fades applied back to the same file
    save_audio(base_prep_192k32b_path, base_prep_192k32b_data, 192000)

    # print(f"Audio saved with fades applied to {base_prep_192k32b_path}")

if __name__ == "__main__":
    run()

