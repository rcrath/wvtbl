
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
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy import stats
from pydub import AudioSegment
from scipy.interpolate import interp1d
import warnings

# Check if the correct number of command-line arguments are provided
if len(sys.argv) >= 2:
    start_file = sys.argv[1]  # The name of the start file, e.g., 'gtrsaw07a_233.wav'
else:
    print("Usage: python wvtbl.py <start_file>.wav")
    sys.exit(1)

# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)

# Define the source folder and construct the full path to the start file
source_folder = "source"
# print(f"DEBUG: source_folder: {source_folder}")
start_file= os.path.join(source_folder, start_file)  # Using the first argument for the start file name
print(f"File to be processed: {start_file}")
# Check if the start file exists within the source folder
if not os.path.exists(start_file):
    print(f"'{start_file}' does not exist in the source folder. Please ensure the file is there and try again.")
    sys.exit(1)

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
# warnings.filterwarnings("ignore", message="Chunk (non-data) not understood, skipping it.")

# Declare amplitude_tolerance_db as a global variable
amplitude_tolerance_db = -60

# Define specific subdirectories inside the base
single_cycles_folder = os.path.join(base, 'single_cycles')
os.makedirs(single_cycles_folder, exist_ok=True)

single_cycles_192k32b = os.path.join(single_cycles_folder, '192k32b')
os.makedirs(single_cycles_192k32b, exist_ok=True)

pwr2_192_2048 = os.path.join(single_cycles_folder, 'pwr2_192_2048')
os.makedirs(pwr2_192_2048, exist_ok=True)

# Define and create the output folder for the 256 frame combined files for Serum wavetables
serum_wavetable_folder = os.path.join(base, 'serum_wavetable')
os.makedirs(serum_wavetable_folder, exist_ok=True)

# serum_2048x256 = os.path.join(serum_wavetable_folder, 'serum_2048x256')
# os.makedirs(serum_2048x256, exist_ok=True)

base_prep_192k32b_path = os.path.join(tmp_folder, base_prep_192k32b)
# print(f"{base_prep_192k32b_path}")

def amplitude_to_db(amplitude):
    # Prevent log of zero or negative values by setting a minimum amplitude level (e.g., 1e-10)
    amplitude[amplitude == 0] = 1e-10
    return 20 * np.log10(abs(amplitude))

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
def is_file_silent(start_file):
    data, _ = sf.read(start_file)  # Read the file
    return np.all(data == 0)  # Check if all values in data are 0

def prompt_for_start_frame(highest_frame):
    while True:
        start_frame_input = input(f"Enter the starting frame (1 to {highest_frame}): ")
        try:
            start_frame = int(start_frame_input)
            if 1 <= start_frame <= highest_frame:
                return start_frame
            else:
                print(f"Please enter a number within the range 1 to {highest_frame}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


# block 2

# pitch detection functions

# Pitch finding using crepe neural net   

# read the audio for use in next, test_crepe
def run_test_crepe(base_prep_192k32b_path):
    sr, audio = wavfile.read(base_prep_192k32b_path)
    time, frequency, confidence = test_crepe(audio, sr)
    return time, frequency, confidence

# test_crepe
def test_crepe(base_prep_192k32b_path):
    sr, audio = wavfile.read(base_prep_192k32b_path)
    time, frequency, confidence, activation = crepe.predict(base_prep_192k32b_data, sr, viterbi=True) 
    return frequency, confidence  # Returns frequency and confidence of prediction

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
    combined_2048x256_frame_out_path = os.path.join(serum_wavetable_folder, combined_2048x256_file_name)
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


# block 3

# begin upsample section
print("\nUpsampling source and fading in and out ...")

# Load the waveform and sample rate from the input file
sample_rate, bit_depth = wavfile.read(start_file)


y, sr = librosa.load(start_file, sr=None)  # Load the file with its original sample rate
# print(f"DEBUG: base_prep_192k32b_path_file: {base_prep_192k32b_path_file}")


# Calculate the duration of the input waveform in seconds
duration_seconds = len(start_file) / sample_rate

# Calculate the number of samples needed to interpolate to 192k while keeping the same duration
target_samples_192k = round(192000 * duration_seconds)

# Resample the input waveform to 192k samples using the best interpolation method
interpolated_input_192k32b = interpolate_best(start_file_data, sample_rate, 192000)
# print(f"DEBUG: base_prep_192k32b: {base_prep_192k32b}")
# print(f"DEBUG: interpolated_input_192k32b: {interpolated_input_192k32b}")


# Save the interpolated input as a temporary wave file
wavfile.write(os.path.join(tmp_folder, base_prep_192k32b), 192000, interpolated_input_192k32b)

# set variables for reading files
base_prep_192k32b_data, sr = librosa.load(os.path.join(tmp_folder, base_prep_192k32b), sr=None)
base_prep_192k32b_data_path = os.path.join(tmp_folder, base_prep_192k32b)
# print(f"base_prep_192k32b_data: {base_prep_192k32b_data}")
# print(f"DEBUG: base_prep_192k32b_data_path: {base_prep_192k32b_data_path}")

# print(f"DEBUG: path: {tmp_folder}/{base_prep_192k32b}")
# print(f"DEBUG: base_prep_192k32b_data: {base_prep_192k32b_data}")


# Add a 5-millisecond fade in and fade out
fade_samples = int(0.001 * 192000)  # 1 milliseconds at 192k samples/second
fade_window = np.linspace(0, 1, fade_samples)

interpolated_input_192k32b = interpolated_input_192k32b.astype(np.float32)
interpolated_input_192k32b[:fade_samples] *= fade_window
interpolated_input_192k32b[-fade_samples:] *= fade_window[::-1]

# Save the faded audio to a new file
wavfile.write("faded_output.wav", 192000, interpolated_input_192k32b)
print(f"Source file upsampled to {base_prep_192k32b_path}")


# upsample ends here...begin pitch section
print(f"\nAI Pitch detect {base_prep_192k32b_path}\n")# Pitch finding using crepe neural net

# Pitch finding using crepe neural net

print("⌄⌄⌄⌄⌄⌄⌄⌄⌄⌄IGNORE THESE WARNINGS⌄⌄⌄⌄⌄⌄⌄⌄⌄⌄")  
frequency_test, confidence_test = test_crepe(base_prep_192k32b_path)

# Define the tolerance range
lower_bound_crepe = 20  # Adjust this lower bound as needed
upper_bound_crepe = 880  # Adjust this upper bound as needed

# Filter frames that meet the criterion within the specified range
filtered_frames_crepe = frequency_test[(frequency_test >= lower_bound_crepe) & (frequency_test <= upper_bound_crepe)]

# Calculate the mode frequency and confidence of the filtered frames
mode_frequency_crepe = np.mean(filtered_frames_crepe)
mode_confidence_crepe = np.mean(confidence_test[(frequency_test >= lower_bound_crepe) & (frequency_test <= upper_bound_crepe)])
mode_frequency_crepe_int = round(mode_frequency_crepe)
mode_confidence_crepe_int = round(mode_confidence_crepe * 100)

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

# Output the results
print("⌃⌃⌃⌃⌃⌃⌃⌃⌃⌃END OF JUNK TO IGNORE⌃⌃⌃⌃⌃⌃⌃⌃⌃⌃\n")
print("\nCREPE neural net pitch detection results:")

if mode_frequency is not None:
    print(f"Mode Frequency: {mode_frequency} Hz (within -49 to +50 cents range)")
    print(f"mode Confidence for Mode Frequency: {round(mode_confidence_avg * 100)}%")
else:
    print("No mode frequency found within the specified range.")

# --- "pitch" section ends here ---


# block 4
# --- begin "segmentation" section ---
print("\nSegmentation underway...")

segment_sizes = []  # Initialize the list to hold segment sizes
prev_start_index = 0  # Start from the beginning
some_small_amplitude = 10**(amplitude_tolerance_db / 20)  # Convert to linear scale
base_prep_192k32b_path_file = os.path.join(tmp_folder, base_prep_192k32b)
samplerate, data = wavfile.read(base_prep_192k32b_path_file)

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
            else:
                print(f"Warning: Empty wave cycle detected at index {i}")

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
    wavfile.write(tmp_base_seg_path, samplerate, wave_cycle)
    #  print(f"Debug: Final segment {segment_index} written: {tmp_base_seg_path}")
    segment_index += 1

print(f"{segment_index -2} segments  in temporary folder.")


# block 5

# --- begin "sort and label segments" ---
print("Sorting and labeling segments...")

# check for short wavecycles
plus_minus_tolerance_percent = 50
plus_minus_tolerance = plus_minus_tolerance_percent / 100
# Initialize a variable to track if any pairs qualify
any_qualify = False

# Initialize wavecycle_samples_target_192 with a default or null value
wavecycle_samples_target_192 = None  # Or some default value if appropriate


try:
    

    # Check if mode_frequency_crepe_int is non-zero before using it to update wavecycle_samples_target to wavecycle_samples_target_192
    if mode_frequency_crepe_int != 0:
        wavecycle_samples_target_192 = round(192000 / mode_frequency_crepe_int)
        print(f"Target wave cycle samples (wavecycle_samples_target_192): {wavecycle_samples_target_192}")
    else:
        print("Warning: Crepe pitch is zero or undetermined. Unable to create wavecycle_samples_target_192.")
except Exception as e:
    mode_frequency_crepe_int = 0
    print(f"Error: {e}")

# Calculate the upper and lower bounds for the number of samples
lower_bound = wavecycle_samples_target_192 * (1 - plus_minus_tolerance)
upper_bound = wavecycle_samples_target_192 * (1 + plus_minus_tolerance)

# Initialize a variable to keep track of how many consecutive segments are within the tolerance
consecutive_count = 0

# Initialize variables to store the indices of the first three qualifying segments
first_three_indices = []
# Initialize the list to store the starting sample index for each segment
segment_start_indices = [0]

# Calculate the cumulative sum of segment sizes to get the starting index for each segment
for size in segment_sizes:
    segment_start_indices.append(segment_start_indices[-1] + size)

# Remove the last entry because it's beyond the last segment
segment_start_indices.pop()


# Iterate through the segment sizes
for idx, size in enumerate(segment_sizes):
    if lower_bound <= size <= upper_bound:
        consecutive_count += 1
        first_three_indices.append(idx)
        
        # Check if you've found three consecutive segments
        if consecutive_count == 3:
            break
    else:
        # Reset if a segment is outside the tolerance
        consecutive_count = 0
        first_three_indices = []

# Check if three consecutive qualifying segments were found
if len(first_three_indices) == 3:
    first_qualifying_idx = first_three_indices[0]

    # Calculate the sample number before the first good segment
    if first_qualifying_idx > 0:
        last_sample_before_good = segment_start_indices[first_qualifying_idx] - 1
        print(f"First three consecutive qualifying segments are at indices: {first_three_indices}")
    else:
        last_sample_before_good = 0  # If the first segment is good, then it starts at the beginning

else:
    print("Did not find three consecutive segments within the tolerance range.")

# block 6
# Rename remaining files outside the tolerance range
print("Renaming remaining files outside the tolerance range...")

# Initialize variables
first_qualifying_idx = None
any_outside_tolerance_renamed = False

# Check if any qualifying segments were found earlier in the script
if len(first_three_indices) == 3:
    first_qualifying_idx = first_three_indices[0]

    # Calculate the sample number before the first good segment
    if first_qualifying_idx > 0:
        last_sample_before_good = segment_start_indices[first_qualifying_idx] - 1
    else:
        last_sample_before_good = 0  # If the first segment is good, then it starts at the beginning

    print(f"Length of attack phase: {last_sample_before_good} samples")
else:
    print("Did not find three consecutive segments within the tolerance range.")

# Loop through all the segments that weren't previously renamed to '_atk'
for i in range(len(segment_sizes)):
    # Skip the segments that were already renamed to '_atk' or outside tolerance range
    if first_qualifying_idx is not None and (i < first_qualifying_idx or i in first_three_indices):
        continue

    # Check if the segment is outside the tolerance
    if segment_sizes[i] < lower_bound or segment_sizes[i] > upper_bound:
        any_outside_tolerance_renamed = True

        # Calculate percent deviation from the target
        deviation_percent = ((segment_sizes[i] - wavecycle_samples_target_192) / wavecycle_samples_target_192) * 100

        # Construct the original segment name and path
        tmp_name = f"{base}_seg_{i:04d}{ext}"
        tmp_file_path = os.path.join(tmp_folder, tmp_name)

        # Construct the new name and path with a label indicating it's deviant
        dev_name = f"{base}_seg_{i:04d}_dev{ext}"
        dev_path = os.path.join(tmp_folder, dev_name)

        # Rename the file and print the deviation information
        if os.path.exists(tmp_file_path):
            os.rename(tmp_file_path, dev_path)
            # print(f"DEBUG: {dev_name} deviates {deviation_percent:.2f}%")

# After checking all segments, report if no files were outside the tolerance range
if not any_outside_tolerance_renamed:
    print("No segment files fall outside the tolerance range.")

# block 7
# ---INTERPOLATION ---
print("\nInterpolating...")
# Initialize the list to store interpolation ratios
interpolation_ratios = []

# Calculate interpolation ratios for each segment
for segment_size in segment_sizes:
    ratio = wavecycle_samples_target_192 / segment_size
    interpolation_ratios.append(ratio)# Iterate through all files in the tmp folder
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
            tmp_file_path = os.path.join(tmp_folder, file)

            # Read the original segment and get its sample rate and subtype
            data, samplerate = sf.read(tmp_file_path)
            info = sf.info(tmp_file_path)

            # Determine the correct subtype for writing based on the subtype of the original file
            if info.subtype in ['PCM_16', 'PCM_24', 'PCM_32']:
                write_subtype = info.subtype
            elif info.subtype == 'FLOAT':
                write_subtype = 'FLOAT'
            elif info.subtype == 'DOUBLE':
                write_subtype = 'FLOAT'
            else:
                print(f"Unsupported subtype: {info.subtype} for file {file}")
                continue  # Skip this file and proceed to the next one

            segment_idx = int(segment_idx_str)
            if segment_idx >= len(interpolation_ratios):
                print(f"Segment index out of range for file {file}")
                continue  # Skip this file and proceed to the next one

            # Use the corresponding interpolation ratio to calculate the target length
            wavecycle_samples_target_192 = int(round(interpolation_ratios[segment_idx] * len(data)))

            # Apply interpolation to adjust the segment length to the target length
            interpolated_segment = interpolate_seg(data, samplerate, wavecycle_samples_target_192)

            # Append the interpolated segment to the list to retain it
            interpolated_segments.append(interpolated_segment)

            # Construct the new file name using the captured parts and save in wavetables folder
            single_cycles_192k32b_name = f"{base}_seg_{segment_idx_str}{suffix}.wav"
            single_cycles_192k32b_path = os.path.join(single_cycles_192k32b, single_cycles_192k32b_name)
    
            # Write the interpolated segment to a new file
            sf.write(single_cycles_192k32b_path, interpolated_segment, samplerate, subtype=write_subtype)

        else:
            print(f"Invalid segment index in filename: {file}")

    else:
        # print(f"DEBUG: File does not match pattern: {file}")
        print("\nChecking Lengths...")

# block 8

# Initialize a list to keep track of files with incorrect lengths and a counter
incorrect_files = []
incorrect_lengths = 0  # Initialize the counter for files with incorrect lengths

# Checking lengths of all segments in the wavetables folder...
for file in os.listdir(single_cycles_192k32b):
    if file.endswith(".wav"):
        # Construct the full path to the file
        file_path = os.path.join(single_cycles_192k32b, file)
        
        # Read the file to get its data
        data, _ = sf.read(file_path)
        
        # Check if the length of the data matches the target
        if len(data) != wavecycle_samples_target_192:
            # print(f"Debug: File {file} has incorrect length: {len(data)} samples (expected {wavecycle_samples_target_192})")
            incorrect_lengths += 1
            incorrect_files.append(file)  # Add the file name to the list

# Print a summary of the test results
total_files_in_single_cycles_192k32b = len([f for f in os.listdir(single_cycles_192k32b) if f.endswith('.wav')])
correct_lengths = total_files_in_single_cycles_192k32b - incorrect_lengths

print(f"\nPost-Interpolation Length Check Summary:")
print(f"Total files checked in wavetables folder: {total_files_in_single_cycles_192k32b}")
print(f"Files with correct length: {correct_lengths}")

if incorrect_lengths == 0:
    print("Success: Every file in the wavetables folder is precisely the expected length.")
else:
    print(f"{incorrect_lengths} of {total_files_in_single_cycles_192k32b} files not equal to target length.")
    for incorrect_file in incorrect_files:
        print(incorrect_file)

# Code to handle files with incorrect lengths
for file in incorrect_files:
    # Construct the full path to the file
    file_path = os.path.join(single_cycles_192k32b, file)
    
    # Read the original file
    data, samplerate = sf.read(file_path)
    original_length = len(data)
    
    # Calculate the new ratio for this specific file based on its length
    new_ratio = wavecycle_samples_target_192 / original_length
    
    # print(f"DEBUG: Processing {file}: Original length: {original_length}, Target length: {wavecycle_samples_target_192}, Ratio: {new_ratio}")
    
    # Interpolate the data to the correct length
    correct_length_data = interpolate_seg(data, samplerate, wavecycle_samples_target_192)
    
    # print(f"DEBUG: Corrected length: {len(correct_length_data)} (Expected: {wavecycle_samples_target_192})")

    # Determine the subtype for writing
    info = sf.info(file_path)
    write_subtype = 'FLOAT' if info.subtype not in ['FLOAT', 'DOUBLE'] else info.subtype
    
    # Write the interpolated data back to the original file name
    sf.write(file_path, correct_length_data, samplerate, subtype=write_subtype)
    # print(f"DEBUG: Wrote corrected data to {file_path}")

# block 9
# --- powers of two ---
print(f"\nChanging single cycles to nearest power of 2 number of samples..")

nearest_192_higher_pwr2 = int(2**np.ceil(np.log2(wavecycle_samples_target_192)))

# serum_pwr2_any = os.path.join(serum_wavetable_folder, f'192k_pwr2_{int(nearest_192_higher_pwr2)}')
# os.makedirs(serum_pwr2_any, exist_ok=True)

# Define the single cycle folder named '192' to save the interpolated segments in
subfolder_192_name = "192"
single_cycles_pwr2_any = os.path.join(single_cycles_folder, f'192k_pwr2_{nearest_192_higher_pwr2}')

# Create the single_cycles_pwr2_any folder if it doesn't exist
if not os.path.exists(single_cycles_pwr2_any):
    os.makedirs(single_cycles_pwr2_any)

# print(f"DEBUG: Source 192 waveforms folder set to: {single_cycles_192k32b}")
# print(f"DEBUG: Resampled waveforms folder set to: {single_cycles_pwr2_any}")

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

# block 10
# --- Combining Specified Number of Wavecycles into a Single File ---
print("\nCombining single cycles to 256 frames ---")

# Use the `single_cycles_192k32b` directory to determine the count for total_files_in_single_cycles_192k32b
total_files_in_single_cycles_192k32b = len([f for f in os.listdir(single_cycles_192k32b) if f.endswith(ext)])

# Specify the number of frames to combine
frames_to_combine_wt = 256  # Adjust this value as needed

# Ensure there are enough frames available
if total_files_in_single_cycles_192k32b >= frames_to_combine_wt:
    highest_start_frame_wt = total_files_in_single_cycles_192k32b - frames_to_combine_wt + 1

    # Prompt the user for the starting frame within the valid range, if more than one choice is available
    if highest_start_frame_wt > 1:
        while True:
            start_frame_wt_input = input(f"Enter the starting frame (1 to {highest_start_frame_wt}): ")
            try:
                start_frame_wt = int(start_frame_wt_input)
                if 1 <= start_frame_wt <= highest_start_frame_wt:
                    break
                else:
                    print(f"Please enter a number within the range 1 to {highest_start_frame_wt}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    else:
        start_frame_wt = 1  # Default to the first frame if only one choice is available

    # Continue with combining frames...
    combined_wt_frames = []
    for filename in sorted(os.listdir(single_cycles_192k32b))[start_frame_wt - 1:start_frame_wt - 1 + frames_to_combine_wt]:
        if filename.endswith(ext):
            file_path = os.path.join(single_cycles_192k32b, filename)
            waveform, sr = sf.read(file_path)
            combined_wt_frames.append(waveform)

    combined_wt_frame = np.concatenate(combined_wt_frames, axis=0)
    combined_wt_frame_out = f"{base}_wt_{start_frame_wt:04d}_{frames_to_combine_wt}_frame_orig_pitch{ext}"
    combined_wt_frame_out_path = os.path.join(serum_wavetable_folder, combined_wt_frame_out)
    sf.write(combined_wt_frame_out_path, combined_wt_frame, sr, subtype='FLOAT')
    print(f"Combined {frames_to_combine_wt} frames starting from frame {start_frame_wt} saved as: {combined_wt_frame_out_path}")

else:
    # Not enough frames to proceed with user input, use backfill method
    print("Not enough frames to complete forward fill. Performing forward-backward fill...")
    # Calculate the number of frames to fill backward
    backward_fill_count = frames_to_combine - total_files_in_single_cycles_192k32b

    # Initialize the list for the combined frames
    combined_frames = []

    # Forward fill with available frames
    for filename in sorted(os.listdir(single_cycles_192k32b))[:total_files_in_single_cycles_192k32b]:
        if filename.endswith(ext):
            file_path = os.path.join(single_cycles_192k32b, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)

    # Backward and inverted fill for the remainder
    for filename in sorted(os.listdir(single_cycles_192k32b))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os.path.join(single_cycles_192k32b, filename)
            waveform, sr = sf.read(file_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)

    # Concatenate all frames along the first axis (time axis)
    combined_frame = np.concatenate(combined_frames, axis=0)

    # Define the output file name
    backfill_file_name = f"{base}_combined_backfilled_{frames_to_combine}_frames{ext}"

    # Construct the full path for the output file
    backfill_file_path = os.path.join(serum_wavetable_folder, backfill_file_name)

    # Write the combined frames to the output file
    sf.write(backfill_file_path, combined_frame, sr, subtype='FLOAT')

    print(f"Backfilled combined file created at {backfill_file_path}")

# block 11
print("\nCreating standard 2048x256 Serum Wavetable...")

# Define the subfolders for original and resampled waveforms
target_wt192_length = 2048  # Serum standard wt target length for all files

# Define and create the output folder for resampled waveforms
# serum_wavetable_folder = os.path.join(serum_wavetable_folder, 'serum_wavetable')
os.makedirs(serum_wavetable_folder, exist_ok=True)

# Define and create the subfolder within 'pwr2' named 'wt192'
# pwr2_192_2048 = os.path.join(single_cycles_folder, pwr2_192_2048)
os.makedirs(pwr2_192_2048, exist_ok=True)
print(f"DEBUG: pwr2_192_2048 set to: {pwr2_192_2048}")
print(f"DEBUG: Source 192 waveforms folder set to: {single_cycles_192k32b}")
print(f"DEBUG: Resampled waveforms folder set to: {pwr2_192_2048}")

# Calculate the ratio of the 2048 to wavecycle_samples_target_192
pwr_of_2_wt192_ratio = target_wt192_length / wavecycle_samples_target_192
print(f"interpolation ratio to get to next highest power of two: {round(pwr_of_2_wt192_ratio, 2)}")

# Calculate the target length for wt192K files as the power of 2 target
pwr_of_2_wt192_target = int(round(wavecycle_samples_target_192 * pwr_of_2_wt192_ratio))

for filename in os.listdir(single_cycles_192k32b):
    if filename.endswith('.wav'):
        wvtbl_192_path = os.path.join(single_cycles_192k32b, filename)
        data, original_sr = sf.read(wvtbl_192_path)
        interpolated_data = interpolate_seg(data, original_sr, target_wt192_length)
        output_wt192_file_path = os.path.join(pwr2_192_2048, filename)
        sf.write(output_wt192_file_path, interpolated_data, original_sr, subtype='FLOAT')

# Combining Specified Number of Wavecycles into a Single File
print("combining 2048 sample by 256 frame serum wavetable as wav file...")
frames_to_combine_wt = 256  # 256 is the wt for serum standard
wt192_count_remainder = total_files_in_single_cycles_192k32b - frames_to_combine_wt

# print(f"DEBUG: Available Frames = {total_files_in_single_cycles_192k32b}")
# print(f"DEBUG: frames to fill (ignore if negative)  = {wt192_count_remainder}")

if total_files_in_single_cycles_192k32b == 0:
    print("ERROR: No frames available to combine. Please check the source directory.")
    exit()

elif total_files_in_single_cycles_192k32b == frames_to_combine_wt:
    print("2048 frames available. Proceeding with combination...")
    combined_wt192_frames = []
    for filename in sorted(os.listdir(pwr2_192_2048)):
        if filename.endswith(ext):
            file_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_path)
            combined_wt192_frames.append(waveform)

    combined_wt192_frame = np.concatenate(combined_wt192_frames, axis=0)
    combined_wt192_frame_out = f"{base}_wt192_full_frame{ext}"
    combined_wt192_frame_out_path = os.path.join(pwrs2_wt192_folder, combined_wt192_frame_out)
    sf.write(combined_wt192_frame_out_path, combined_wt192_frame, sr, subtype='FLOAT')
    print(f"Combined full frame saved as: {combined_wt192_frame_out_path}")

elif total_files_in_single_cycles_192k32b > frames_to_combine_wt:
    # print(f"DEBUG: total_files_in_single_cycles_192k32b: {total_files_in_single_cycles_192k32b}, frames_to_combine_wt: {frames_to_combine_wt}")
    highest_start_frame_wt192 = max(1, total_files_in_single_cycles_192k32b - frames_to_combine_wt)
    if highest_start_frame_wt192 > 1:
        # Set the starting frame
        start_frame_wt192 = start_frame_wt
        
        # Calculate the ending frame based on the selected starting frame
        ending_frame_wt192 = start_frame_wt192 + frames_to_combine_wt - 1

        # Initialize the list for the combined frames
        combined_192k_256_frames = []

        # Iterate through the files in the source directory starting from the selected frame
        for filename in sorted(os.listdir(pwr2_192_2048))[start_frame_wt192 - 1:ending_frame_wt192]:
            if filename.endswith(ext):
                # Construct the full path for the file
                file_path = os.path.join(pwr2_192_2048, filename)
                
                # Read the waveform data
                waveform, sr = sf.read(file_path)
                
                # Append the waveform data to the combined_192k_256_frames list
                combined_192k_256_frames.append(waveform)

        # Concatenate all frames along the first axis (time axis)
        combined_192k_2048_frame_wt192k = np.concatenate(combined_192k_256_frames, axis=0)

        # Define the output file name with the starting frame included
        combined_192k_2048_frame_out = f"{base}_serum_start{start_frame_wt192:04d}_2048x{frames_to_combine_wt}{ext}"

        # Construct the full path for the output file
        combined_2048x256_frame_out_path = os.path.join(serum_wavetable_folder, combined_192k_2048_frame_out)

        # Write the combined frames to the output file
        sf.write(combined_2048x256_frame_out_path, combined_192k_2048_frame_wt192k, sr, subtype='FLOAT')

        print(f"Combined {frames_to_combine_wt} frames starting from frame {start_frame_wt192} saved as: {combined_2048x256_frame_out_path}")

else:
    # Not enough frames to complete forward fill. Performing forward-backward fill...
    print("Not enough frames to complete forward fill. Performing forward-backward fill...")
    backward_fill_count = frames_to_combine_wt - total_files_in_single_cycles_192k32b

    # Initialize the list for the combined frames
    combined_frames = []

    # Forward fill with available frames
    for filename in sorted(os.listdir(pwr2_192_2048))[:total_files_in_single_cycles_192k32b]:
        if filename.endswith(ext):
            file_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)

    # Backward and inverted fill for the remainder
    for filename in sorted(os.listdir(pwr2_192_2048))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os

    # Forward fill with available frames
    for filename in sorted(os.listdir(pwr2_192_2048))[:total_files_in_single_cycles_192k32b]:
        if filename.endswith(ext):
            file_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)
    
    # Backward and inverted fill for the remainder
    for filename in sorted(os.listdir(pwr2_192_2048))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os.path.join(pwr2_192_2048, filename)
            waveform, sr = sf.read(file_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)
    
    # Concatenate all frames along the first axis (time axis)
    combined_frame = np.concatenate(combined_frames, axis=0)

    # Define the output file name
    backfill_file_name = f"{base}_combined_backfilled_{frames_to_combine_wt}_frames{ext}"

    # Construct the full path for the output file
    combined_2048x256_frame_out_path = os.path.join(serum_wavetable_folder, backfill_file_name)

    # Write the combined frames to the output file
    sf.write(combined_2048x256_frame_out_path, combined_frame, sr, subtype='FLOAT')

    print(f"Backfilled combined file created at {combined_2048x256_frame_out_path}")

print(f"\n\n\n\nDONE\n\n\n\n")

print("\n --- PAUSE ---")



# print(f"DIRS: {dir()}  # \n a dictionary of local variables\n")
# print(f"GLOBALS: {globals()}  # a dictionary of global variables\n")
# print(f"LCL: {locals()}  # a dictionary of local variables\n")
# print(f"VARs: {vars()}  # a dictionary of local variables\n")


