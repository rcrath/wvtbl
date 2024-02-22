# block 1 
import os
import sys
import wave
import crepe
import librosa
import re
import shutil
import resampy
import subprocess
import math
from math import log2, isclose
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy import stats
from pydub import AudioSegment
from scipy.interpolate import interp1d
import warnings
from datetime import datetime
import time
import threading

# Define the source folder
source_folder = "source"

# Function to list .wav files and allow user selection
def list_and_select_wav_files(source_folder):
    # List all wav files in the source_folder
    files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]

    # Sort files alphabetically in a case-insensitive manner
    files.sort(key=lambda x: x.lower())
    
    # Display files to the user with an index
    for i, file in enumerate(files):
        print(f"{i+1}: {file}")
    
    # Optionally, add an option to quit the script at the end of the file list
    print("\nEnter the number of the file to select, or type 'q' to exit.")

    # Get user input and handle selection or quit
    selection = input("Selection: ").strip()
    if selection.lower() == 'q':
        print("Quitting script.")
        sys.exit()  # Exit the script entirely
    
    try:
        # Convert the user's input into an index and retrieve the corresponding file name
        selected_index = int(selection) - 1  # Adjust for zero-based indexing
        if 0 <= selected_index < len(files):
            return files[selected_index]  # Return the selected file name
        else:
            print("Invalid selection. Please try again.")
            return list_and_select_wav_files(source_folder)  # Recursive call to retry
    except ValueError:
        print("Please enter a valid number or 'q'.")
        return list_and_select_wav_files(source_folder)  # Recursive call to retry

# Decide how to select the start file based on the presence of command-line arguments
if len(sys.argv) >= 2:
    start_file_name = sys.argv[1]  # Use the command-line argument if provided
else:
    # Call the function to list and let the user select a wav file if no command-line argument is given
    start_file_name = list_and_select_wav_files(source_folder)

# Construct full path to the start file
start_file = os.path.join(source_folder, start_file_name)

# Ensure the file exists
if not os.path.exists(start_file):
    print(f"'{start_file}' does not exist. Please check the file name and try again.")
    sys.exit(1)

print(f"Processing file: {start_file}")

# Load the waveform and sample rate from the input file
sample_rate, start_file_data = wavfile.read(start_file)

# Generate the output filename without the extension
base = os.path.splitext(os.path.basename(start_file))[0]

# create base folder
os.makedirs(base, exist_ok=True)  # 'base' is a provided variable from the command line
# print(f"DEBUG: base: {base}")
# Set the file extension
ext = ".wav"


# define tmpfile for upsampled full wavfile
base_prep_192k32b = f"{base}-prep_192k32{ext}"

# Load the waveform and sample rate from the input file
sample_rate, start_file_data = wavfile.read(start_file)

# Create the "tmp" folder if it doesn't exist
tmp_folder = os.path.join(base, 'tmp')
os.makedirs(tmp_folder, exist_ok=True)

# Filter out the specific warning...this is not working.  
warnings.filterwarnings("ignore", message="Chunk (non-data) not understood, skipping it.")

# Declare amplitude_tolerance_db as a global variable
amplitude_tolerance_db = -60

# Define specific subdirectories inside the base
single_cycles_folder = os.path.join(base, 'single_cycles')
os.makedirs(single_cycles_folder, exist_ok=True)

single_cycles_192k32b = os.path.join(single_cycles_folder, '192k32b')
os.makedirs(single_cycles_192k32b, exist_ok=True)

# Define and create the output folder for the 256 frame combined files for Serum wavetables
concat_folder = os.path.join(base, 'concat')
os.makedirs(concat_folder, exist_ok=True)

base_prep_192k32b_path = os.path.join(tmp_folder, base_prep_192k32b)
# print(f"{base_prep_192k32b_path}")


# Function to calculate rising zero crossings in a waveform
def is_rising_zero_crossing(data, index):
    # Ensure index is within the valid range
    if index <= 0 or index >= len(data) - 1:  # -1 to handle the end of the file
        return False
    
    # Check for a rising zero crossing: previous sample is negative, and the current one is positive
    if data[index - 1] < 0 and data[index] >= 0:
        return True
    
    return False

def is_full_wavecycle(segment):
    global amplitude_tolerance_db  # Use the global variable

    if len(segment) < 3:
        return False

    # Convert the first and last samples to dB
    first_sample_db = 20 * np.log10(max(abs(segment[0]), 1e-10))
    last_sample_db = 20 * np.log10(max(abs(segment[-1]), 1e-10))

    # Check if the first and last samples are near zero in dB
    if first_sample_db > amplitude_tolerance_db or last_sample_db > amplitude_tolerance_db:
        return False

    # Detect zero crossings
    zero_crossings = np.where(np.diff(np.signbit(segment)))[0]

    # Ensure there's at least one significant zero crossing
    if len(zero_crossings) < 2:
        return False

    return True

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
        
        # print(f"Saved '{output_file_name}' with {len(segment)} samples")

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
        # print(f"Saved '{output_file_name}' with {len(last_segment)} samples")


# function to upsample or downsample files.
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
    return resampy.resample(waveform, original_sr, target_sr)

# function to interpolate segments by ratios
def interpolate_seg(data, original_sr, wavecycle_samples_target_192):
    #print(f"original_sr, {original_sr} kHz,  wavecycle_samples_target_192 {wavecycle_samples_target_192} samples")
    """
    Interpolate a waveform segment to a specified target length.
    
    Parameters:
    - data: np.ndarray, the input waveform (audio segment).
    - original_sr: int, the original sample rate of the waveform.
    - wavecycle_samples_target_192: int, the desired number of samples in the output.
    
    Returns:
    - np.ndarray, the resampled waveform segment with the specified target length.
    """
    original_length = len(data)
    target_sample_rate = int(round(wavecycle_samples_target_192 * original_sr / original_length))
    interpolated = resampy.resample(data, original_sr, target_sample_rate)

    # Ensure the interpolated segment is the exact target length (trim or pad if necessary)
    if len(interpolated) > wavecycle_samples_target_192:
        # Trim excess samples
        interpolated = interpolated[:wavecycle_samples_target_192]
    elif len(interpolated) < wavecycle_samples_target_192:
        # Pad with zeros to reach the target length
        padding = np.zeros(wavecycle_samples_target_192 - len(interpolated))
        interpolated = np.concatenate((interpolated, padding))

    return interpolated
# Function to check if a file is effectively silent (zero amplitude throughout)

def adjust_sample_rate_based_on_wavecycle_length(input_file_path, target_length, output_folder):
    data, original_sr = sf.read(input_file_path)
    target_sr = original_sr  # Default to original sample rate

    # Adjust the sample rate based on the target wavecycle length
    if 3072 < target_length < 6145:
        target_sr = 96000
    elif 6148 < target_length < 9600:
        target_sr = 48000

    # Resample the audio if the target sample rate differs from the original
    if target_sr != original_sr:
        data_resampled = resampy.resample(data, original_sr, target_sr)
        new_filename = f"{os.path.splitext(os.path.basename(input_file_path))[0]}_{target_sr}.wav"
        output_file_path = os.path.join(output_folder, new_filename)
        sf.write(output_file_path, data_resampled, target_sr)
        # print(f"Resampled {os.path.basename(input_file_path)} to {target_sr}Hz, saved as {new_filename}.")

# block 2

# pitch detection functions

# Pitch finding using crepe neural net   

# test_crepe
def test_crepe(base_prep_192k32b_path):
    sr, audio = wavfile.read(base_prep_192k32b_path)
    time, frequency, confidence, activation = crepe.predict(base_prep_192k32b_data, sr, viterbi=True) 
    return frequency, confidence  # Returns frequency and confidence of prediction

def mark_attack_segments(first_qualifying_idx, base, tmp_folder, ext):
    any_attack_phase_renamed = False
    for i in range(first_qualifying_idx):
        seg_name = f"{base}_seg_{i:04d}{ext}"
        seg_file_path = os.path.join(tmp_folder, seg_name)
        atk_name = f"{base}_seg_{i:04d}_atk{ext}"
        atk_path = os.path.join(tmp_folder, atk_name)
        if os.path.exists(seg_file_path):
            os.rename(seg_file_path, atk_path)
            any_attack_phase_renamed = True
    return any_attack_phase_renamed

def mark_deviant_segments(segment_sizes, lower_bound, upper_bound, wavecycle_samples_target_192, base, tmp_folder, ext):
    outside_tolerance_files = []
    for i, segment_size in enumerate(segment_sizes):
        if segment_size < lower_bound or segment_size > upper_bound:
            seg_name = f"{base}_seg_{i:04d}{ext}"
            seg_file_path = os.path.join(tmp_folder, seg_name)
            dev_name = f"{base}_seg_{i:04d}_dev{ext}"
            dev_path = os.path.join(tmp_folder, dev_name)
            if os.path.exists(seg_file_path):
                os.rename(seg_file_path, dev_path)
                outside_tolerance_files.append(dev_name)
    return outside_tolerance_files

def frequency_to_note(frequency, A4=440):
    """
    Maps a given frequency to the nearest equal temperament note.
    A4 is set to 440 Hz by default.
    """
    if frequency <= 0:
        return None
    # Calculate the number of half steps away from A4
    half_steps = 12 * np.log2(frequency / A4)
    # Round to the nearest half step to get the equal temperament note
    nearest_half_step = int(round(half_steps))
    # Convert back to frequency
    nearest_frequency = A4 * 2 ** (nearest_half_step / 12)
    return nearest_frequency

def note_to_frequency(note):
    """
    Convert a musical note to its corresponding frequency.
    Assumes A4 = 440Hz as standard tuning.
    """
    A4 = 440
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    octave = int(note[-1])  # Extract the octave number
    note_name = note[:-1]  # Extract the note name (without octave)
    
    if note_name in notes:
        # Calculate the note's index in the octave from C0 up to the note
        note_index = notes.index(note_name) - notes.index('A') + (octave - 4) * 12
        # Calculate the frequency
        return A4 * (2 ** (note_index / 12))
    else:
        print("\n\33[33mInvalid note name.\33[0m")
        return None

def frequency_to_note_and_cents(frequency, A4=440):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    c0 = A4 * pow(2, -4.75)
    half_steps_above_c0 = round(12 * log2(frequency / c0))
    note = notes[half_steps_above_c0 % 12]
    octave = half_steps_above_c0 // 12
    exact_frequency = c0 * pow(2, half_steps_above_c0 / 12)
    cents = round(1200 * log2(frequency / exact_frequency))
    return f"{note}{octave}", cents

def get_manual_frequency_input(lowest_freq, highest_freq):
    """
    Prompt for a frequency in Hz or a musical note. Returns the frequency in Hz,
    or None if the user decides to skip by pressing Enter.
    """
    while True:
        user_input = input(f"Enter the frequency in Hz (between {lowest_freq}Hz and {highest_freq}Hz), a musical note (e.g., A4, C#3), or press Enter to skip: ").strip()

        if not user_input:  # User presses Enter to skip
            return None

        if user_input.replace('.', '', 1).isdigit():  # Input is in Hz
            freq = float(user_input)
            if lowest_freq <= freq <= highest_freq:
                return freq
            else:
                print(f"Frequency out of bounds. Please enter a value between {lowest_freq}Hz and {highest_freq}Hz.")
        else:  # Input is potentially a note
            freq = note_to_frequency(user_input)
            if freq and lowest_freq <= freq <= highest_freq:
                return freq
            elif not freq:
                print(f"Invalid note. Please enter a valid musical note (e.g., A4, C#3).")
            else:
                print(f"Note frequency out of bounds. Please enter a note corresponding to a frequency between {lowest_freq}Hz and {highest_freq}Hz.")


def initialize_settings():
    # Initial settings with default values
    settings = {
        'freq_note_input': 'enter',  # Default action is to proceed without setting
        'percent_tolerance': 5,  # Default tolerance percent
        'discard_atk_choice': 'N',  # Default choice for discarding attack segments
        'discard_dev_choice': 'N',  # Default choice for discarding deviant segments
        'discard_good_choice': 'N',  # Default choice for discarding good segments
        'cleanup_choice': 'Y',  # Default choice for cleanup

    }
    return settings

def update_settings(settings):
    accept_defaults = input("\n\n    Accept all defaults? (Y/n, default=Y): ").strip().upper() or 'Y'
    if accept_defaults == 'Y':
        print("Proceeding with defaults.\n\n\n")
        return settings

    # Update settings only if not accepting defaults
    settings['freq_note_input'] = input(f"     Enter the frequency in Hz (between {lowest_freq}Hz and {highest_freq}Hz), \n     Or note (no flats) with octave \n     (e.g., A3, A#3, B3, C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4), \n     or press <enter> to proceed without setting it.\n\033[36mHz, Note, or <enter>: \033[0m").strip() or 'enter'
    
    # Update the tolerance setting based on user input
    percent_input = input(f"Set deviation tolerance from target length (default={settings['percent_tolerance']}%): ").strip()
    if percent_input:
        try:
            # Ensure the input is converted to float for percentage calculation later
            settings['percent_tolerance'] = float(percent_input)
            print(f"Tolerance updated to {settings['percent_tolerance']}%.")
        except ValueError:
            print("Invalid input. Proceeding with default deviation tolerance.")

    settings['discard_atk_choice'] = input("Discard attack segments? (y/N, default=N): ").strip().upper() or 'N'
    settings['discard_dev_choice'] = input("Discard deviant segments? (y/N, default=N): ").strip().upper() or 'N'
    settings['discard_good_choice'] = input("Discard good segments? (y/N, default=N): ").strip().upper() or 'N'
    settings['cleanup_choice'] = input("Perform cleanup? (Y/n, default=Y): ").strip().upper() or 'Y'
    
    return settings
def is_segment_deviant(index, base, tmp_folder, ext):
    """
    Check if the given segment index corresponds to a deviant segment.
    
    Parameters:
    - index: The index of the segment to check.
    - base: The base name for the files.
    - tmp_folder: The temporary folder where the segments are stored.
    - ext: The file extension of the segments.
    
    Returns:
    - True if the segment is tagged as deviant, False otherwise.
    """
    # Construct the expected filename for a deviant segment
    deviant_file_name = f"{base}_seg_{index:04d}_dev{ext}"
    deviant_file_path = os.path.join(tmp_folder, deviant_file_name)
    
    # Return True if the deviant file exists, False otherwise
    return os.path.exists(deviant_file_path)
def spinner():
    global stop_spinner
    spinner_chars = ["-", "\\", "|", "/"]
    idx = 0
    while not stop_spinner:
        sys.stdout.write('\r' + spinner_chars[idx % len(spinner_chars)])
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    # Clear the spinner before exiting
    sys.stdout.write('\r')
    sys.stdout.flush()

import numpy as np
import soundfile as sf

def normalize_audio_to_peak(file_path, target_peak=-6):
    """
    Normalize the audio file to a target peak in dBFS.

    Parameters:
    - file_path: Path to the audio file.
    - target_peak: Target peak level in dBFS.
    """
    data, samplerate = sf.read(file_path)
    peak = np.max(np.abs(data))
    if peak == 0:
        return  # Avoid division by zero if the audio is silent
    normalization_factor = 10 ** ((target_peak - 20 * np.log10(peak)) / 20)
    normalized_data = data * normalization_factor
    sf.write(file_path, normalized_data, samplerate)

# Main script starts here
lowest_freq = 20
highest_freq = 880

settings = initialize_settings()  # Initialize with defaults
settings = update_settings(settings)  # Optionally update settings

# Initialize a variable at the script's start to track if the tolerance was manually set
tolerance_manually_set = False

if settings.get('accept_defaults', 'Y').lower() != 'y':
    # This block is where you've determined that the user does not want to accept all defaults
    # Now, prompt for the tolerance percentage as part of the initial setup
    user_input = input(f"Enter deviation tolerance percentage (default={settings['percent_tolerance']}%): ").strip()
    if user_input:
        try:
            settings['percent_tolerance'] = float(user_input)
            tolerance_manually_set = True  # Mark that the tolerance was manually set
        except ValueError:
            print("Invalid input. Using default deviation tolerance.")
else:
    # If accepting all defaults, there's no need to change 'tolerance_manually_set' as it remains False
    print("Proceeding with all defaults, including the default deviation tolerance.")

# Now, `settings` contains the final values to use
# print(settings)  # Demonstration of settings; replace with actual use in your script


# Initialize freq_est to 0 by default to handle cases where no frequency is provided
# Assuming settings have been initialized and updated as needed
freq_est = None
freq_est_manually_set = False

# Check if frequency/note input is part of the settings and act accordingly
if settings.get('freq_note_input') and settings['freq_note_input'] != 'enter':
    freq_input = settings['freq_note_input']
    if freq_input.replace('.', '', 1).isdigit():
        freq_est = float(freq_input)
    else:
        freq_est = note_to_frequency(freq_input)  # Assuming note_to_frequency is defined and available
    if freq_est:
        freq_est_manually_set = True
else:
    # Optionally call get_manual_frequency_input() here if you want to prompt for input even when settings don't specify a frequency
    pass

# If a valid frequency was entered, you can proceed with further processing
if freq_est:
    note_est, cents = frequency_to_note_and_cents(freq_est)  # Ensure you have this function defined
    cents_format = f"+{cents}" if cents > 0 else f"{cents}"
    note_est_cents = f"{note_est} {cents_format}"
    print(f"\nFrequency entered: {freq_est}Hz")
    print(f"Corresponding note and deviation: {note_est_cents} cents\n")


 # block 3a

# begin upsample section
# print("\nUpsampling source and adding fade in and out ...")
# print("\n⌄⌄⌄⌄⌄⌄⌄⌄⌄⌄IGNORE ANY WARNINGS⌄⌄⌄⌄⌄⌄⌄⌄⌄⌄") 
# Load the waveform and sample rate from the input file
sample_rate, bit_depth = wavfile.read(start_file)


y, sr = librosa.load(start_file, sr=None)  # Load the file with its original sample rate
# print(f"DEBUG: base_prep_192k32b_path: {base_prep_192k32b_path}")


# Calculate the duration of the input waveform in seconds
duration_seconds = librosa.get_duration(y=y, sr=sr)
# print(f"DEBUG: duration_seconds: {duration_seconds}, \n       len(y): {len(y)}")

# Calculate the number of samples needed to interpolate to 192k while keeping the same duration
target_samples_192k = round(192000 * duration_seconds)
# print(f"DEBUG: target_samples_192k: {target_samples_192k}")
# Resample the input waveform to 192k samples using the best interpolation method
interpolated_input_192k32b = interpolate_best(start_file_data, sample_rate, 192000)
# print(f"DEBUG: base_prep_192k32b: {base_prep_192k32b}")
# print(f"DEBUG: interpolated_input_192k32b: {interpolated_input_192k32b}")


# Save the interpolated input as a temporary wave file
wavfile.write(os.path.join(tmp_folder, base_prep_192k32b), 192000, interpolated_input_192k32b)

# set variables for reading files
base_prep_192k32b_data, sr = librosa.load(os.path.join(tmp_folder, base_prep_192k32b), sr=None)
base_prep_192k32b_path = os.path.join(tmp_folder, base_prep_192k32b)
# print(f"base_prep_192k32b_data: {base_prep_192k32b_data}")
# print(f"DEBUG: base_prep_192k32b_path: {base_prep_192k32b_path}")

# print(f"DEBUG: path: {tmp_folder}/{base_prep_192k32b}")
# print(f"DEBUG: base_prep_192k32b_path: {base_prep_192k32b_path}")

sample_rate = 192000 

# Define the number of samples for fade in and fade out
fade_samples = 2048  # Adjust this as needed

# Ensure the audio data is long enough for the fades
if len(base_prep_192k32b_data) > 2 * fade_samples:
    # Create the fade-in and fade-out windows
    fade_in_window = np.linspace(0, 1, fade_samples, dtype=np.float32)
    fade_out_window = np.linspace(1, 0, fade_samples, dtype=np.float32)

    # Apply the fade in to the beginning of the audio data
    base_prep_192k32b_data[:fade_samples] *= fade_in_window

    # Apply the fade out to the end of the audio data
    base_prep_192k32b_data[-fade_samples:] *= fade_out_window
else:
    print("Audio data too short for the specified fade length.")

# Save the audio data with fades applied back to a new file, maintaining the 32-bit float format
# output_path = os.path.join(tmp_folder, base_prep_192k32b)  # Adjust as needed to save in the desired location
sf.write(base_prep_192k32b_path, base_prep_192k32b_data, sample_rate, subtype='FLOAT')

# print(f"Audio saved with fades applied to {base_prep_192k32b_path}")
# print(f"sample_rate: {sample_rate}")

# block 3b
# begin pitch section

# print(f"\nAI Pitch detect {base_prep_192k32b}\n")# Pitch finding using crepe neural net

# Pitch finding using crepe neural net

print("⌄⌄⌄⌄⌄⌄⌄⌄⌄⌄IGNORE THESE WARNINGS⌄⌄⌄⌄⌄⌄⌄⌄⌄⌄")  
frequency_test, confidence_test = test_crepe(base_prep_192k32b_path)

# Output the results
print("⌃⌃⌃⌃⌃⌃⌃⌃⌃⌃END OF JUNK TO IGNORE⌃⌃⌃⌃⌃⌃⌃⌃⌃⌃\n\n\n")


# Define the tolerance range
lower_bound_crepe = lowest_freq  # Adjust this lower bound as needed
upper_bound_crepe = highest_freq  # Adjust this upper bound as needed

# Prepare frequencies for mode calculation
filtered_frequencies = [frequency for frequency in frequency_test if lower_bound_crepe <= frequency <= upper_bound_crepe]
mapped_frequencies = [frequency_to_note(f) for f in filtered_frequencies]

if len(mapped_frequencies) > 0:
    mode_result = stats.mode(mapped_frequencies)
    if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
        mode_frequency = mode_result.mode[0]
    else:
        mode_frequency = mode_result.mode  # mode is a scalar

    # Filter confidences corresponding to the mode frequency
    mode_confidences = [confidence_test[i] for i, f in enumerate(filtered_frequencies) if frequency_to_note(f) == mode_frequency]

    # Calculate the mode confidence for the mode frequency
    mode_confidence_avg = np.mean(mode_confidences) if len(mode_confidences) > 0 else 0
else:
    print("No frequencies to process.")
    mode_frequency = None
    mode_confidence_avg = 0

confidence_threshold = 0.50

# Assume mode_frequency and mode_confidence_avg have been determined as before

if freq_est_manually_set:
    # If a manual frequency was previously set, offer a choice between it and the detected mode frequency
    print(f"Detected mode frequency: {round(mode_frequency)} Hz with confidence {round(mode_confidence_avg * 100)}%.")
    choice_prompt = f"Choose frequency source: \n1. Detected mode ({mode_frequency} Hz), \n2. Manually entered ({freq_est} Hz) \n[default: 1]: "
    user_choice = input(choice_prompt).strip()

    if user_choice == '2':
        # User chooses the manually entered frequency
        mode_frequency = freq_est
        print(f"Using manually entered frequency: {freq_est} Hz")
    else:
        # User chooses the detected mode frequency or does not provide valid input; no change needed
        print(f"Proceeding with detected mode frequency: {mode_frequency} Hz with confidence {round(mode_confidence_avg * 100)}%.")
elif mode_confidence_avg < confidence_threshold:
    # If no manual frequency was set and confidence is low, prompt for manual frequency input
    print(f"Pitch detection results are not reliable, {round(mode_confidence_avg * 100)}%")
    freq_est = get_manual_frequency_input(lowest_freq, highest_freq)
    if freq_est is not None:
        # User chooses to manually set the frequency
        mode_frequency = freq_est
        freq_est_manually_set = True
        print(f"Using manually entered frequency: {freq_est} Hz.")
    else:
        print("No manual frequency input provided. Unable to proceed.")
#else:
    # If confidence is high and no manual frequency was previously set, proceed with detected mode frequency
    # print(f"Frequency: {mode_frequency} Hz, {round(mode_confidence_avg * 100)}% confidence.")

# Ensure wavecycle_samples_target_192 is calculated and valid
wavecycle_samples_target_192 = round(192000 / mode_frequency) if mode_frequency else None

if not wavecycle_samples_target_192 or wavecycle_samples_target_192 <= 0:
    print("Unable to proceed without a valid target wave cycle sample count.")
    sys.exit(1)


# --- "pitch" section ends here ---


# block 4
# --- begin "segmentation" section ---
# print("\nSegmentation underway...")

segment_sizes = []  # Initialize the list to hold segment sizes
prev_start_index = 0  # Start from the beginning
some_small_amplitude = 10**(amplitude_tolerance_db / 20)  # Convert to linear scale
base_prep_192k32b_path = os.path.join(tmp_folder, base_prep_192k32b)
data, samplerate = sf.read(base_prep_192k32b_path)

# Process the first segment explicitly if the start is near zero
if abs(data[0]) < some_small_amplitude:  # Define 'some_small_amplitude' based on your fades
    for i in range(1, len(data)):
        if is_rising_zero_crossing(data, i):
            prev_start_index = i
            break  # Found the first real zero crossing, move to normal processing

# Variables to hold the first two segments temporarily
first_segment = None
second_segment = None
segment_index = 0

# Read the audio data and sampling rate from the file

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
                # Debug: Print the segment's properties
                if segment_index == 0 or segment_index == 1:
                    # Print the length of the segment
                    #  print(f"Debug: Segment {segment_index} length: {len(wave_cycle)} samples")
                    # Print the amplitude at the beginning of the segment in dB
                    #  print(f"Debug: Segment {segment_index} start amplitude: -{-20 * np.log10(max(abs(wave_cycle[0]), 1e-10))} dB")
                    # Print the amplitude at the end of the segment in dB
                    #  print(f"Debug: Segment {segment_index} end amplitude: -{-20 * np.log10(max(abs(wave_cycle[-1]), 1e-10))} dB")

                    # Check if this is the first or second segment
                    if segment_index == 0:
                        first_segment = wave_cycle
                    elif segment_index == 1:
                        second_segment = wave_cycle

                # Write segment to file if it's not the first or second segment
                if segment_index > 1 or segment_index == 0:
                    base_seg = f"{base}_seg_{segment_index:04d}{ext}"
                    tmp_base_seg_path = os.path.join(tmp_folder, base_seg)
                    wavfile.write(tmp_base_seg_path, samplerate, wave_cycle)
                    #  print(f"Debug: Segment {segment_index} written: {tmp_base_seg_path}")

                segment_index += 1
            # else:
                # print(f"Warning: Empty wave cycle detected at index {i}")

# Check if the first two segments contain full wave cycles
if first_segment is not None and second_segment is not None:
    full_cycle_first = is_full_wavecycle(first_segment)
    full_cycle_second = is_full_wavecycle(second_segment)

    # Debug: Print the evaluation of the first two segments
    #  print(f"Debug: First segment is a full cycle: {full_cycle_first}")
    #  print(f"Debug: Second segment is a full cycle: {full_cycle_second}")

    if not full_cycle_first or not full_cycle_second:
        # Combine the first two segments
        combined_segment = np.concatenate((first_segment, second_segment))

        # Write the combined segment to the '0001' file
        combined_path = os.path.join(tmp_folder, f"{base}_seg_0001{ext}")
        wavfile.write(combined_path, samplerate, combined_segment)
        #  print(f"Debug: Combined segment written: {combined_path}")

        # Delete the '0000' file if it exists
        first_path = os.path.join(tmp_folder, f"{base}_seg_0000{ext}")
        if os.path.exists(first_path):
            os.remove(first_path)
            #  print(f"Debug: Deleted: {first_path}")
    else:
        # If both segments are full cycles, write them out as normal
        for i, segment in enumerate([first_segment, second_segment], start=0):
            segment_path = os.path.join(tmp_folder, f"{base}_seg_{i:04d}{ext}")
            wavfile.write(segment_path, samplerate, segment)
            #  print(f"Debug: Segment {i} written: {segment_path}")

# Handle the last segment
if prev_start_index < len(data):
    wave_cycle = data[prev_start_index:]
    # Check if the wave cycle is full before writing to file
    if is_full_wavecycle(wave_cycle):
        wavfile.write(tmp_base_seg_path, samplerate, wave_cycle)
        # print(f"Debug: Final segment {segment_index} written: {tmp_base_seg_path}")
        segment_index += 1
    # else:
        # print(f"Debug: Final segment {segment_index} is not a full wave cycle and was not written.")

# Block 5 - Handling Tolerance and Identifying Qualifying Segments

# Assign tolerance from settings and calculate as decimal
plus_minus_tolerance_percentage = settings['percent_tolerance']
plus_minus_tolerance = plus_minus_tolerance_percentage / 100.0

# Check and set wavecycle target based on mode_frequency
if mode_frequency and mode_frequency > 0:
    wavecycle_samples_target_192 = round(192000 / mode_frequency)
else:
    print("Mode frequency not determined. Unable to define wavecycle target.")
    sys.exit(1)  # Exit if mode frequency is undefined

# Recalculate bounds with updated tolerance
lower_bound = mode_frequency * (1 - plus_minus_tolerance)
upper_bound = mode_frequency * (1 + plus_minus_tolerance)

print(f"Tolerance (+/-): {round(plus_minus_tolerance_percentage)}%, {round(lower_bound)}Hz to {round(upper_bound)}Hz, {round(upper_bound - lower_bound)}Hz range")

# Initialize list for storing non-deviant segment indices
non_deviant_segments = []

# Identify non-deviant segments within tolerance
for idx, size in enumerate(segment_sizes):
    if lower_bound <= size <= upper_bound and not is_segment_deviant(idx, base, tmp_folder, ext):
        non_deviant_segments.append(idx)

# Find first set of three consecutive non-deviant segments
first_set_start_index = None
for i in range(len(non_deviant_segments) - 2):
    if non_deviant_segments[i] + 1 == non_deviant_segments[i + 1] and non_deviant_segments[i + 1] + 1 == non_deviant_segments[i + 2]:
        first_set_start_index = non_deviant_segments[i]
        # print(f"First three consecutive non-deviant segments start at index: {first_set_start_index}")
        break

# Rename preceding segments as attack phase, if applicable
if first_set_start_index is not None:
    for i in range(first_set_start_index):
        seg_name = f"{base}_seg_{i:04d}{ext}"
        seg_file_path = os.path.join(tmp_folder, seg_name)
        atk_name = f"{base}_seg_{i:04d}_atk{ext}"
        atk_path = os.path.join(tmp_folder, atk_name)
        if os.path.exists(seg_file_path):
            os.rename(seg_file_path, atk_path)
    # print("Attack phase segments marked.")
else:
    print("No consecutive non-deviant segments for attack phase marking.")

# block 6 rename atk and dev, count

# Rename attack phase files and remaining files outside the tolerance range
# print("Renaming attack phase and remaining files outside the tolerance range...")

# Initialize variables
first_qualifying_idx = None  # Use None to indicate that no qualifying index has been found yet
any_outside_tolerance_renamed = False
any_attack_phase_renamed = False



# Initialize first_three_indices as an empty list
first_three_indices = []

# Below is where you prompt for user input or use defaults
if settings.get('accept_defaults', 'Y').lower() != 'y':
    user_input = input(f"How far can single cycles deviate from the target length? (default={percent_tolerance}%) \nPercentage: ").strip()
    if user_input:
        try:
            plus_minus_tolerance_percentage = float(user_input)
            plus_minus_tolerance = plus_minus_tolerance_percentage / 100.0
        except ValueError:
            print("Invalid input. Using default deviation tolerance.")
            plus_minus_tolerance = plus_minus_tolerance_percentage / 100.0
else:
    plus_minus_tolerance = plus_minus_tolerance_percentage / 100.0

# Recalculate the lower and upper bounds here
if wavecycle_samples_target_192:
    lower_bound = wavecycle_samples_target_192 * (1 - plus_minus_tolerance)
    upper_bound = wavecycle_samples_target_192 * (1 + plus_minus_tolerance)
else:
    print("Wavecycle target not set, cannot proceed.")
    # Here, handle the case appropriately, such as exiting or setting a default


if len(first_three_indices) >= 3:
    first_qualifying_idx = first_three_indices[0]
    any_attack_phase_renamed = mark_attack_segments(first_qualifying_idx, base, tmp_folder, ext)
    print("Attack phase files renamed." if any_attack_phase_renamed else "No attack phase files to rename.")

# Now handle deviant segments based on tolerance
outside_tolerance_files = mark_deviant_segments(segment_sizes, lower_bound, upper_bound, wavecycle_samples_target_192, base, tmp_folder, ext)
# print(f"Count of files outside the tolerance range: {len(outside_tolerance_files)}" if outside_tolerance_files else "No segment files fall outside the tolerance range.")




# block 7
# ---INTERPOLATION ---
print("\nInterpolating...\n")
# Flag to control the spinner

stop_spinner = False

# Start the spinner thread
spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()


# Initialize the list to store interpolation ratios
interpolation_ratios = []

# Create a new subfolder '192k32b_singles' within 'tmp_folder'
singles_folder = os.path.join(tmp_folder, '192k32b_singles')
os.makedirs(singles_folder, exist_ok=True)

# Calculate interpolation ratios for each segment
for segment_size in segment_sizes:
    ratio = wavecycle_samples_target_192 / segment_size
    interpolation_ratios.append(ratio)

# Iterate through all files in the tmp folder
interpolated_segments = []  # List to retain all the interpolated segments
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
            if segment_idx >= len(interpolation_ratios):
                print(f"Segment index out of range for file {file}")
                continue  # Skip this file and proceed to the next one

            # Use the corresponding interpolation ratio to calculate the target length
            target_length = int(round(interpolation_ratios[segment_idx] * len(data)))

            # Apply interpolation to adjust the segment length to the target length
            interpolated_segment = interpolate_seg(data, samplerate, target_length)

            # Append the interpolated segment to the list to retain it
            interpolated_segments.append(interpolated_segment)

            # Construct the new file name using the captured parts
            single_cycles_192k32b_name = f"{base}_seg_{segment_idx_str}{suffix}.wav"
            single_cycles_192k32b_path = os.path.join(singles_folder, single_cycles_192k32b_name)
    
            # Write the interpolated segment to the '192k32b_singles' folder
            sf.write(single_cycles_192k32b_path, interpolated_segment, samplerate, subtype=write_subtype)

        # else:
            # print(f"Invalid segment index in filename: {file}")
    # else:
        # print("\nChecking Lengths...")




# No longer directly prompt the user in block 8, but use the choices from 'settings'
discard_atk_choice = settings['discard_atk_choice']
discard_dev_choice = settings['discard_dev_choice']
discard_good_choice = settings['discard_good_choice']

# Iterate through all files in the '192k32b_singles' folder
for file in os.listdir(singles_folder):
    if file.endswith(".wav"):
        # Construct the full path to the file
        file_path = os.path.join(singles_folder, file)

        # Check if file is an '_atk', '_dev', or 'good' file and handle according to user's choice
        if '_atk' in file and discard_atk_choice == 'Y':
            # print("Discarding attack files")
            os.remove(file_path)  # Make sure to actually delete/discarding the file if needed
            continue  # Skip to the next file
        elif '_dev' in file and discard_dev_choice == 'Y':
            # print("Discarding deviant files")
            os.remove(file_path)  # Make sure to actually delete/discarding the file if needed
            continue  # Skip to the next file
        elif not ('_atk' in file or '_dev' in file) and discard_good_choice == 'Y':
            # print("Discarding 'good' files")
            os.remove(file_path)  # Make sure to actually delete/discarding the file if needed
            continue  # Skip to the next file

        # If the file is not discarded based on the user's choices:
        # Read the original file
        data, samplerate = sf.read(file_path)
        original_length = len(data)

        # Interpolate the data to the correct length if necessary
        correct_length_data = interpolate_seg(data, samplerate, wavecycle_samples_target_192)

        # Construct the path for saving to 'single_cycles_192k32b'
        all_same_length__path = os.path.join(single_cycles_192k32b, file)
        
        # Write the interpolated data to the 'single_cycles_192k32b' folder
        sf.write(all_same_length__path, correct_length_data, samplerate, subtype='FLOAT')
        # Optionally print a message indicating the file has been processed and saved

# print(f"Processed files according to user preferences and saved to 'single_cycles_192k32b'")

# Initialize counters for each type of segment
total_segments_count = 0
attack_segments_count = 0
good_segments_count = 0
deviant_segments_count = 0


# Iterate through all files in the 'singles_folder' to count before any potential discarding
for file in os.listdir(singles_folder):
    if file.endswith(".wav"):
        total_segments_count += 1
        if '_atk' in file:
            attack_segments_count += 1
        elif '_dev' in file:
            deviant_segments_count += 1
        else:
            good_segments_count += 1
    if file.endswith(".wav"):
        # Construct the full path to the file
        file_path = os.path.join(singles_folder, file)

# Determine actions based on user choices
atk_action = "discarding" if discard_atk_choice == 'Y' else "keeping"
good_action = "discarding" if discard_good_choice == 'Y' else "keeping"
dev_action = "discarding" if discard_dev_choice == 'Y' else "keeping"

# Stop the spinner and print your statement
stop_spinner = True
spinner_thread.join()  # Wait for the spinner to finish

# Print counts before discarding with action
# print(f"Tolerance percentage: {settings['percent_tolerance']}%")
print(f"Total segments before discarding: {total_segments_count}")
print(f"Total attack segments: {attack_segments_count}, {atk_action}")
print(f"Total good segments: {good_segments_count}, {good_action}")
print(f"Total deviant segments: {deviant_segments_count}, {dev_action}")

print("\nConstructing wavetables ...")

# Flag to control the spinner

stop_spinner = False

# Start the spinner thread
spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()


# block 9
# --- powers of two ---
# print(f"--- BEGIN POWERS OF 2 ---")

nearest_192_higher_pwr2 = int(2**np.ceil(np.log2(wavecycle_samples_target_192)))

# Define the single cycle folder named '192' to save the interpolated segments in
subfolder_192_name = "192"
single_cycles_pwr2_any = os.path.join(single_cycles_folder, f'192k_pwr2_{nearest_192_higher_pwr2}')

# Create the single_cycles_pwr2_any folder if it doesn't exist
if not os.path.exists(single_cycles_pwr2_any):
    os.makedirs(single_cycles_pwr2_any)

# print(f"Source 192 waveforms folder set to: {single_cycles_192k32b}")
# print(f"Resampled waveforms folder set to: {single_cycles_pwr2_any}")

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

# Block 9.1 - Transforming to Fixed Length of 2048 Samples
# print("\nTransforming to fixed length of 2048 samples...")

# Define the target length
target_length_2048 = 2048

# Define and create the pwr2_192_2048 directory
pwr2_192_2048 = os.path.join(single_cycles_folder, 'pwr2_192_2048_94hz')
if not os.path.exists(pwr2_192_2048):
    os.makedirs(pwr2_192_2048)
    # print(f"Created directory: {pwr2_192_2048}")
# else:
    # print(f"Directory already exists: {pwr2_192_2048}")

# Process files from single_cycles_pwr2_any
for filename in os.listdir(single_cycles_pwr2_any):
    if filename.endswith('.wav'):
        input_file_path = os.path.join(single_cycles_pwr2_any, filename)
        data, original_sr = sf.read(input_file_path)
        interpolated_data = interpolate_seg(data, original_sr, target_length_2048)
        output_file_path = os.path.join(pwr2_192_2048, filename)
        sf.write(output_file_path, interpolated_data, original_sr, subtype='FLOAT')

# print("Finished processing files for {pwr2_192_2048}.")


# block 9.2 Adjusting Sample Rate for Low Sample Lengths

output_folder_for_adjusted_samples = (single_cycles_pwr2_any) # Define your output folder path here

# Ensure the output folder exists
if not os.path.exists(output_folder_for_adjusted_samples):
    os.makedirs(output_folder_for_adjusted_samples)

for filename in os.listdir(single_cycles_pwr2_any):
    if filename.endswith('.wav'):
        input_file_path = os.path.join(single_cycles_pwr2_any, filename)
        # Call the function with wavecycle_samples_target_192 as the target_length
        # Here, you can directly pass wavecycle_samples_target_192 if it's a fixed value for all files,
        # or calculate it based on the content of each file if necessary
        adjust_sample_rate_based_on_wavecycle_length(input_file_path, wavecycle_samples_target_192, output_folder_for_adjusted_samples)

output_folder = output_folder_for_adjusted_samples

# Flag to track if any file with specific endings exists
five_digit_ending_exists = False

# First, check if there are any files with the specific endings
for filename in os.listdir(output_folder):
    if re.search(r'_(48000|96000)\.wav$', filename):
        five_digit_ending_exists = True
        print("\nAdjusting Sample Rate for Low Sample Lengths...")
        break  # Break the loop if at least one matching file is found

# If files with specific endings exist, proceed to check and delete files without those endings
if five_digit_ending_exists:
    for filename in os.listdir(output_folder):
        if filename.endswith('.wav') and not re.search(r'_(48000|96000)\.wav$', filename):
            # Delete files that do not end with '48000.wav' or '96000.wav'
            os.remove(os.path.join(output_folder, filename))
            

# print("Sample rate adjustment and cleanup completed.")
        
# print("Sample rate adjustment completed.")

# block 10 concatenate

# Concatenate ALL files in single_cycles_pwr2_any
all_frames_pwr2 = []
for filename in sorted(os.listdir(single_cycles_pwr2_any)):
    if filename.endswith('.wav'):
        file_path = os.path.join(single_cycles_pwr2_any, filename)
        data, sr = sf.read(file_path)
        all_frames_pwr2.append(data)

# Concatenate all frames into a single array
wavetable_data_pwr2 = np.concatenate(all_frames_pwr2, axis=0)

# Save the concatenated wavetable
output_file_pwr2 = f"{base}_{nearest_192_higher_pwr2}_all.wav"
output_path_pwr2 = os.path.join(concat_folder, output_file_pwr2)
sf.write(output_path_pwr2, wavetable_data_pwr2, sr, subtype='FLOAT')
# print(f"Saved processed files to '{output_file_pwr2} at {sr}Hz'")

# Concatenate ALL files in pwr2_192_2048
all_frames_2048 = []
for filename in sorted(os.listdir(pwr2_192_2048)):
    if filename.endswith('.wav'):
        file_path = os.path.join(pwr2_192_2048, filename)
        data, sr = sf.read(file_path)
        all_frames_2048.append(data)

# Concatenate all frames into a single array
wavetable_data_2048 = np.concatenate(all_frames_2048, axis=0)

# Save the concatenated wavetable
output_file_2048 = f"{base}_2048_94Hz_all.wav"
output_path_2048 = os.path.join(concat_folder, output_file_2048)
sf.write(output_path_2048, wavetable_data_2048, sr, subtype='FLOAT')
# print(f"Saved processed files to '{output_file_2048}'")

# Assuming {base}_{nearest_192_higher_pwr2}_all.wav and {base}_2048_94Hz_all.wav are located in concat_folder
closest_pwr2_all_path = os.path.join(concat_folder, f"{base}_{nearest_192_higher_pwr2}_all.wav")
hz_94_all_path = os.path.join(concat_folder, f"{base}_2048_94Hz_all.wav")

# block 11
# make wavetables and cleanup

closest_pitch_folder = base
hz_94_folder = base


data_closest_pitch, sr_closest_pitch = sf.read(output_path_pwr2, dtype='float32')
total_samples_closest_pitch = len(data_closest_pitch)

data_94hz, sr_94hz = sf.read(output_path_2048, dtype='float32')
total_samples_94hz = len(data_94hz)

# Now you have the total number of samples for each file
# print(f"Total samples in closest pitch file: {total_samples_closest_pitch}")
# print(f"Total samples in 94Hz file: {total_samples_94hz}")
# For 'closest_pitch' wavetable type
num_full_files_closest_pitch = math.ceil(total_samples_closest_pitch / 524288)
split_and_save_wav_with_correct_padding(closest_pwr2_all_path, closest_pitch_folder, base, "closest_pitch", num_full_files_closest_pitch)

# For '94Hz' wavetable type
num_full_files_94hz = math.ceil(total_samples_94hz / 524288)
split_and_save_wav_with_correct_padding(hz_94_all_path, hz_94_folder, base, "94Hz", num_full_files_94hz)

# Stop the spinner and print your statement
stop_spinner = True
spinner_thread.join()  # Wait for the spinner to finish

# Use the 'concat_folder' variable as the directory where your wavetable files are stored
wavetable_files = [os.path.join(concat_folder, f) for f in os.listdir(concat_folder) if f.endswith('.wav')]

# Normalize each wavetable file in the concat_folder
for wavetable_file in wavetable_files:
    normalize_audio_to_peak(wavetable_file, target_peak=-6)

print(f"All files in '{concat_folder}' have been normalized to peak -6 dBFS.")

def perform_cleanup():

    folders_to_cleanup = [tmp_folder, single_cycles_folder]
    # files_to_cleanup = [output_path_2048, closest_pwr2_all_path]

    try:
        for folder in folders_to_cleanup:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                # print(f"Deleted folder: {folder}")
        '''
        for file_path in files_to_cleanup:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")
        '''
        print("Cleanup completed.")
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

# Check user's choice for cleanup from settings
if settings['cleanup_choice'] == 'Y':
    # print("Starting cleanup...")
    perform_cleanup()
else:
    print("Skipping cleanup as per user choice.")


print(f"\n\n\n\nDONE\n\n\n\n")


'''
print(f"DIRS: {dir()}  # \n a dictionary of local variables\n")
print(f"GLOBALS: {globals()}  # a dictionary of global variables\n")
print(f"LCL: {locals()}  # a dictionary of local variables\n")
print(f"VARs: {vars()}  # a dictionary of local variables\n")
'''