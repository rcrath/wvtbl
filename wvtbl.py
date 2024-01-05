# block 1 
import os
import sys
import wave
import aubio
import librosa
import re
import shutil
import resampy
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.interpolate import interp1d
# Declare amplitude_tolerance_db as a global variable
amplitude_tolerance_db = -60

# Store initial values of variables
# initial_values = locals().copy()

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
def interpolate_seg(data, original_sr, target_length):
    #print(f"original_sr, {original_sr} kHz,  target_length {target_length} samples")
    """
    Interpolate a waveform segment to a specified target length.
    
    Parameters:
    - data: np.ndarray, the input waveform (audio segment).
    - original_sr: int, the original sample rate of the waveform.
    - target_length: int, the desired number of samples in the output.
    
    Returns:
    - np.ndarray, the resampled waveform segment with the specified target length.
    """
    original_length = len(data)
    target_sample_rate = int(round(target_length * original_sr / original_length))
    interpolated = resampy.resample(data, original_sr, target_sample_rate)

    # Ensure the interpolated segment is the exact target length (trim or pad if necessary)
    if len(interpolated) > target_length:
        # Trim excess samples
        interpolated = interpolated[:target_length]
    elif len(interpolated) < target_length:
        # Pad with zeros to reach the target length
        padding = np.zeros(target_length - len(interpolated))
        interpolated = np.concatenate((interpolated, padding))

    return interpolated

# block 2
# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)

# Function to run SoX and get pitch information
def get_pitch_using_sox(base_prep, tmp_folder="tmp"):
    try:
        # Construct the path to the prep file
        base_prep_path_file = os.path.join(tmp_folder, base_prep)

        # Run SoX on the prep file and extract the frequency estimate
        command = f"sox {base_prep_path_file} -n stat 2>&1 | grep 'frequency' | tr -cd '0-9.'"
        result = os.popen(command).read()

        return float(result) if result else 0
    except Exception as e:
        print(f"Error in get_pitch_using_sox: {e}")
        return 0

# Function to check if a file is effectively silent (zero amplitude throughout)
def is_file_silent(file_path):
    data, _ = sf.read(file_path)  # Read the file
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

def combine_and_save_frames(start_frame, frame_count):
    combined_frames = []
    ending_frame = start_frame + frame_count - 1

    for filename in sorted(os.listdir(powersof2_wt192_folder))[start_frame - 1:ending_frame]:
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)

    combined_frame = np.concatenate(combined_frames, axis=0)
    output_file_name = f"{base}_wt192k24b_start{start_frame:04d}_{frame_count}_frame{ext}"
    output_file_path = os.path.join(pwrs2_folder, output_file_name)
    sf.write(output_file_path, combined_frame, sr, subtype='FLOAT')
    print(f"Combined {frame_count} frames starting from frame {start_frame} saved as: {output_file_path}")

def perform_backfill_and_invert():
    backward_fill_count = frames_to_combine_wt - total_files_in_wt192_folder
    combined_frames = []

    for filename in sorted(os.listdir(powersof2_wt192_folder))[:total_files_in_wt192_folder]:
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)
    
    for filename in sorted(os.listdir(powersof2_wt192_folder))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)
    
    combined_frame = np.concatenate(combined_frames, axis=0)
    output_file_name = f"{base}_combined_backfilled_{frames_to_combine_wt}_frames{ext}"
    output_file_path = os.path.join(pwrs2_folder, output_file_name)
    sf.write(output_file_path, combined_frame, sr, subtype='FLOAT')
    print(f"Backfilled combined file created at {output_file_path}")

# Check if command-line arguments are provided
if len(sys.argv) >= 4:
    start_file = sys.argv[1]
    base = sys.argv[2]
    freq_est = float(sys.argv[3])
else:
    print("Usage: python script.py start_file base freq_est")
    sys.exit(1)

prev_start_index = 0  # Start from the beginning
print("Starting segmentation...")
some_small_value = 0.001  # Define 'some_small_value' based on your fades
segment_sizes = []

# Create the "tmp" folder if it doesn't exist
tmp_folder = "tmp"
create_folder(tmp_folder)

# Create the "wavetables" folder if it doesn't exist
wavetables_folder = "wavetables"
create_folder(wavetables_folder)

# Set the file extension
ext = ".wav"

# block 3
# begin upsample section
print("\nUpsample section")

# Load the waveform and sample rate from the input file
sample_rate, start_file_wave = wavfile.read(start_file)

# Calculate wavecycle_samples_target_in based on the provided frequency estimate
wavecycle_samples_float = sample_rate / freq_est
wavecycle_samples_target = round(wavecycle_samples_float)

# Calculate the duration of the input waveform in seconds
duration_seconds = len(start_file_wave) / sample_rate

# Calculate the number of samples needed to interpolate to 192k while keeping the same duration
target_samples_192k = round(192000 * duration_seconds)

# Resample the input waveform to 192k samples using the best interpolation method
interpolated_input = interpolate_best(start_file_wave, sample_rate, 192000)

# Add a 5-millisecond fade in and fade out
fade_samples = int(0.001 * 192000)  # 1 milliseconds at 192k samples/second
fade_window = np.linspace(0, 1, fade_samples)

interpolated_input[:fade_samples] *= fade_window
interpolated_input[-fade_samples:] *= fade_window[::-1]

# Save the interpolated input as a temporary wave file
base_prep = f"{base}-prep{ext}"
wavfile.write(os.path.join(tmp_folder, base_prep), 192000, np.array(interpolated_input, dtype=np.int32))

# Continue with pitch detection using get_pitch_using_sox function
# Initialize wavecycle_samples_target_192 with a default or null value
wavecycle_samples_target_192 = None  # Or some default value if appropriate

# Calculate pitch using SoX
try:
    sox_pitch = get_pitch_using_sox(base_prep)

    # Check if sox_pitch is non-zero before using it to update wavecycle_samples_target to wavecycle_samples_target_192
    if sox_pitch != 0:
        print(f"Pitch calculated using SoX: {sox_pitch} Hz")
        wavecycle_samples_target_192 = round(192000 / sox_pitch)
        print(f"Target wave cycle samples (wavecycle_samples_target_192): {wavecycle_samples_target_192}")
    else:
        print("Warning: SoX pitch is zero or undetermined. Unable to create wavecycle_samples_target_192.")
except Exception as e:
    sox_pitch = 0
    print(f"Error: {e}")


# --- "upsample" section ends here ---

# --- begin "segmentation" section ---
print("\nSegmentation section")


prev_start_index = 0  # Start from the beginning
some_small_value = 0.001  # Example value, adjust as needed
base_prep_path_file = os.path.join(tmp_folder, base_prep)
samplerate, data = wavfile.read(base_prep_path_file)

# Process the first segment explicitly if the start is near zero
if abs(data[0]) < some_small_value:  # Define 'some_small_value' based on your fades
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
                    # print(f"Debug: Segment {segment_index} length: {len(wave_cycle)}")
                    # print(f"Debug: Segment {segment_index} start dB: {-20 * np.log10(max(abs(wave_cycle[0]), 1e-10))}")
                    # print(f"Debug: Segment {segment_index} end dB: {-20 * np.log10(max(abs(wave_cycle[-1]), 1e-10))}")

                    # Check if this is the first or second segment
                    if segment_index == 0:
                        first_segment = wave_cycle
                    elif segment_index == 1:
                        second_segment = wave_cycle

                # Write segment to file if it's not the first or second segment
                if segment_index > 1 or segment_index == 0:
                    base_seg = f"{base}_seg_{segment_index:04d}{ext}"
                    full_path = os.path.join(tmp_folder, base_seg)
                    wavfile.write(full_path, samplerate, wave_cycle)
                    # print(f"Debug: Segment {segment_index} written: {full_path}")

                segment_index += 1
            else:
                print(f"Warning: Empty wave cycle detected at index {i}")

# block 4
# Check if the first two segments contain full wave cycles
if first_segment is not None and second_segment is not None:
    full_cycle_first = is_full_wavecycle(first_segment)
    full_cycle_second = is_full_wavecycle(second_segment)

    # Debug: Print the evaluation of the first two segments
    # print(f"Debug: First segment is full cycle: {full_cycle_first}")
    # print(f"Debug: Second segment is full cycle: {full_cycle_second}")

    if not full_cycle_first or not full_cycle_second:
        # Combine the first two segments
        combined_segment = np.concatenate((first_segment, second_segment))
        
        # Write the combined segment to the '0001' file
        combined_path = os.path.join(tmp_folder, f"{base}_seg_0001{ext}")
        wavfile.write(combined_path, samplerate, combined_segment)
        # print(f"Debug: Combined segment written: {combined_path}")

        # Delete the '0000' file if it exists
        first_path = os.path.join(tmp_folder, f"{base}_seg_0000{ext}")
        if os.path.exists(first_path):
            os.remove(first_path)
            # print(f"Debug: Deleted: {first_path}")
    else:
        # If both segments are full cycles, write them out as normal
        for i, segment in enumerate([first_segment, second_segment], start=0):
            segment_path = os.path.join(tmp_folder, f"{base}_seg_{i:04d}{ext}")
            wavfile.write(segment_path, samplerate, segment)
            # print(f"Debug: Segment {i} written: {segment_path}")
# Handle the last segment
if prev_start_index < len(data):
    wave_cycle = data[prev_start_index:]
    base_seg = f"{base}_seg_{segment_index:04d}{ext}"
    full_path = os.path.join(tmp_folder, base_seg)
    wavfile.write(full_path, samplerate, wave_cycle)
    # print(f"Debug: Final segment {segment_index} written: {full_path}")
    segment_index += 1

print(f"Segmentation complete. {segment_index} segments were created and saved in {tmp_folder}.")

# --- begin "sort and label segments" ---
print("Starting sorting and labeling segments.")

# check for short wavecycles
plus_minus_tolerance_percent = 5
plus_minus_tolerance = plus_minus_tolerance_percent / 100
# Initialize a variable to track if any pairs qualify
any_qualify = False


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
    else:
        last_sample_before_good = 0  # If the first segment is good, then it starts at the beginning

    print(f"length of attack phase: {last_sample_before_good} samples")
    # print(f"First three consecutive qualifying segments are at indices: {first_three_indices}")

    # Rest of your code for renaming files...
else:
    print("Did not find three consecutive segments within the tolerance range.")

# block 5
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
        original_name = f"{base}_seg_{i:04d}{ext}"
        original_path = os.path.join(tmp_folder, original_name)

        # Construct the new name and path with a label indicating it's deviant
        new_name = f"{base}_seg_{i:04d}_dev{ext}"
        new_path = os.path.join(tmp_folder, new_name)

        # Rename the file and print the deviation information
        if os.path.exists(original_path):
            os.rename(original_path, new_path)
            print(f"{new_name} deviates {deviation_percent:.2f}%")

# After checking all segments, report if no files were outside the tolerance range
if not any_outside_tolerance_renamed:
    print("No segment files fall outside the tolerance range.")

# block 6
# ---INTERPOLATION ---
print("begin interpolation")
# Initialize the list to store interpolation ratios
interpolation_ratios = []

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
            original_path = os.path.join(tmp_folder, file)

            # Read the original segment and get its sample rate and subtype
            data, samplerate = sf.read(original_path)
            info = sf.info(original_path)

            # Determine the correct subtype for writing based on the subtype of the original file
            if info.subtype in ['PCM_16', 'PCM_24', 'PCM_32']:
                write_subtype = info.subtype  # Directly use the read subtype
            elif info.subtype == 'FLOAT':
                write_subtype = 'FLOAT'  # For 32-bit float data
            elif info.subtype == 'DOUBLE':
                write_subtype = 'FLOAT'  # Convert 64-bit float (DOUBLE) to 32-bit float for wider compatibility
            else:
                print(f"Unsupported subtype: {info.subtype} for file {file}")
                continue  # Skip this file and proceed to the next one

            # Convert the segment index to an integer
            # Convert the segment index to an integer
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

            # Construct the new file name using the captured parts and save in wavetables folder
            new_name = f"{base}_seg_{segment_idx_str}{suffix}.wav"
            new_path = os.path.join(wavetables_folder, new_name)
    
            # Write the interpolated segment to a new file
            sf.write(new_path, interpolated_segment, samplerate, subtype=write_subtype)
            # print(f"Debug: Interpolated file written: {new_path}")
            
            segment_idx = int(segment_idx_str)

            if segment_idx >= len(interpolation_ratios):
                print(f"Segment index out of range for file {file}")
                continue  # Skip this file and proceed to the next one

            # Use the corresponding interpolation ratio to calculate the target length
            target_length = int(round(interpolation_ratios[segment_idx] * len(data)))

            # Apply interpolation to adjust the segment length to the target length
            interpolated_segment = interpolate_seg(data, samplerate, target_length)

            # Construct the new file name using the captured parts and save in wavetables folder
            new_name = f"{base}_seg_{segment_idx_str}{suffix}.wav"
            new_path = os.path.join(wavetables_folder, new_name)
            
            # Write the interpolated segment to a new file
            sf.write(new_path, interpolated_segment, samplerate, subtype=write_subtype)
            # print(f"Debug: Interpolated file written: {new_path}")
        else:
            print(f"Invalid segment index in filename: {file}")
    else:
        print(f"File does not match pattern: {file}")

else:
        print(f"File does not match pattern: {file}")

# Initialize a list to keep track of files with incorrect lengths and a counter
incorrect_files = []
incorrect_lengths = 0  # Initialize the counter for files with incorrect lengths

# Checking lengths of all segments in the wavetables folder...
for file in os.listdir(wavetables_folder):
    if file.endswith(".wav"):
        # Construct the full path to the file
        file_path = os.path.join(wavetables_folder, file)
        
        # Read the file to get its data
        data, _ = sf.read(file_path)
        
        # Check if the length of the data matches the target
        if len(data) != wavecycle_samples_target_192:
            print(f"Debug: File {file} has incorrect length: {len(data)} samples (expected {wavecycle_samples_target_192})")
            incorrect_lengths += 1
            incorrect_files.append(file)  # Add the file name to the list

# Print a summary of the test results with a modified statement at the end
total_files_in_wavetables = len([f for f in os.listdir(wavetables_folder) if f.endswith('.wav')])
correct_lengths = total_files_in_wavetables - incorrect_lengths

print(f"\nPost-Interpolation Length Check Summary:")
print(f"Total files checked in wavetables folder: {total_files_in_wavetables}")
print(f"Files with correct length: {correct_lengths}")

if incorrect_lengths == 0:
    print("Success: Every file in the wavetables folder is precisely the expected length of {wavecycle_samples_target_192} samples.")
else:
    print(f"{incorrect_lengths} of {total_files_in_wavetables} not equal to target {wavecycle_samples_target_192} samples.")
    # print("Debug: files to resize:")
    for incorrect_file in incorrect_files:
        print(incorrect_file)

# Code to handle files with incorrect lengths
for file in incorrect_files:
    # Construct the full path to the file
    file_path = os.path.join(wavetables_folder, file)
    
    # Read the original file
    data, samplerate = sf.read(file_path)
    original_length = len(data)
    
    # Calculate the new ratio for this specific file based on its length
    new_ratio = wavecycle_samples_target_192 / original_length
    
    print(f"Processing {file}:")
    print(f"  Original length: {original_length}, Target length: {wavecycle_samples_target_192}, Ratio: {new_ratio}")
    
    # Interpolate the data to the correct length at the target sample rate of 192kHz
    correct_length_data = interpolate_seg(data, samplerate, wavecycle_samples_target_192)
    
    print(f"  Corrected length: {len(correct_length_data)} (Expected: {wavecycle_samples_target_192})")

    # Determine the subtype to use for writing
    info = sf.info(file_path)
    write_subtype = 'PCM_24'  # Default to 'PCM_24' if needed, or use info.subtype
    
    # Write the interpolated data back to the original file name
    sf.write(file_path, correct_length_data, samplerate, subtype=write_subtype)
    print(f"  Wrote corrected data to {file_path}")


# block 7
# Paths for the new directories inside wavetables_folder
path_192k32b = os.path.join(wavetables_folder, '192k32b')
path_48k24b = os.path.join(wavetables_folder, '48k24b')

# Create the 192k32b folder
if not os.path.exists(path_192k32b):
    os.makedirs(path_192k32b, exist_ok=True)
    print(f"Created folder: {path_192k32b}")

# Create the 48k24b folder
if not os.path.exists(path_48k24b):
    os.makedirs(path_48k24b, exist_ok=True)
    print(f"Created folder: {path_48k24b}")

# Downsample all files in the wavetables_folder to 48k 24 bits
target_sr = 48000  # Target sample rate for downsampling
for file in os.listdir(wavetables_folder):
    if file.endswith('.wav'):
        original_path = os.path.join(wavetables_folder, file)

        # Load the file
        data, sr = librosa.load(original_path, sr=None)  # Load with original sample rate

        # Resample the audio to 48k
        data_48k = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

        # Extract the four-digit segment number and any suffix
        parts = file.replace(base, '').replace(ext, '').split('_')  # Split the filename into parts
        segment_number = parts[2]  # The four-digit segment number (assuming 'base_seg_NNNN' format)
        suffix = '_' + '_'.join(parts[3:]) if len(parts) > 3 else ''  # Include any suffix like '_atk' or '_dev'

        # Construct new file path with the desired format: base_48k24b_NNNN[_atk or _dev].wav
        new_file_name = f"{base}_48k24b_{segment_number}{suffix}{ext}"
        new_path = os.path.join(wavetables_folder, '48k24b', new_file_name)

        # Write the resampled data to the new file with 24-bit depth
        sf.write(new_path, data_48k, target_sr, subtype='PCM_24')

        # print(f"DEBUG: Downsampled and saved: {file} as {new_file_name}")

print(f"Downsampled and saved to 48k24b")   



# Rename and move original files to 192k32b folder
print("Renaming and moving original files to 192k32b folder...")

for file in os.listdir(wavetables_folder):
    if file.startswith(base) and file.endswith(ext) and '_seg_' in file:
        original_path = os.path.join(wavetables_folder, file)

        # Extract the four-digit segment number and any suffix
        parts = file.replace(base, '').replace(ext, '').split('_')  # Split the filename into parts
        segment_number = parts[2]  # The four-digit segment number (assuming 'base_seg_NNNN' format)
        suffix = '_' + '_'.join(parts[3:]) if len(parts) > 3 else ''  # Include any suffix like '_atk' or '_dev'

        # Construct new file name with the desired format: base_192k32b_NNNN[_atk or _dev].wav
        new_file_name = f"{base}_192k32b_{segment_number}{suffix}{ext}"
        new_path = os.path.join(wavetables_folder, '192k32b', new_file_name)

        # Move the file to the 192k32b folder with the new name
        os.rename(original_path, new_path)
        # print(f"Debug: Moved and renamed: {file} to {new_file_name}")

print("Original files successfully renamed and moved to 192k32b")

# block 8
# --- powers of two ---
print(f"--- BEGIN POWERS OF 2 ---")

print("192k")
# Define the subfolders for original and resampled waveforms
wvtbls192 = os.path.join(wavetables_folder, '192k32b')  # Use the actual variable from uvar if it's different

# Define the output folder for resampled waveforms
pwrs2_folder = os.path.join(wavetables_folder, 'powersof2')  # Adjust the subfolder name as needed

# Create the pwrs2_folder if it doesn't exist
if not os.path.exists(pwrs2_folder):
    os.makedirs(pwrs2_folder)

# Define the subfolder within 'wavetables/powersof2' named '192' and to save the interpolated segments
subfolder_192_name = "192"

powersof2_192_folder = os.path.join(wavetables_folder, "powersof2", subfolder_192_name)

# Create the pwrs2 192 _folder if it doesn't exist
if not os.path.exists(powersof2_192_folder):
    os.makedirs(powersof2_192_folder)


print(f"Source 192 waveforms folder set to: {wvtbls192}")
print(f"Resampled waveforms folder set to: {powersof2_192_folder}")

# Calculate the nearest higher power of 2
nearest_192_higher_power_of_2 = 2**np.ceil(np.log2(wavecycle_samples_target_192))

# Define the ratio of the nearest higher power of 2 to wavecycle_samples_target_192
pwr_of_2_192_ratio = nearest_192_higher_power_of_2 / wavecycle_samples_target_192

# Calculate the target length for 192K files as the power of 2 target
pwr_of_2_192_target = int(round(wavecycle_samples_target_192 * pwr_of_2_192_ratio))


# Calculate the new target length for all 192k files
target_192_length = int(round(wavecycle_samples_target_192 * pwr_of_2_192_ratio))



for filename in os.listdir(wvtbls192):
    if filename.endswith('.wav'):  # Check if the file is a .wav file
        # Construct the full path for the input file using wvtbls192
        input_file_path = os.path.join(wvtbls192, filename)
        
        # Read the waveform and its sample rate
        data, original_sr = sf.read(input_file_path)
        
        # Use interpolate_seg to resample the segment to the new target length
        interpolated_data = interpolate_seg(data, original_sr, target_192_length)
        
        # Construct the full path for the output file using powersof2_192_folder
        output_192_file_path = os.path.join(powersof2_192_folder, filename)
        
        # Save the interpolated segment to the powersof2_192_folder
        sf.write(output_192_file_path, interpolated_data, original_sr, subtype='FLOAT')
        #print(f"Processed and saved: {filename} to {powersof2_192_folder}")


nearest_192_higher_power_of_2 = 2**np.ceil(np.log2(wavecycle_samples_target_192))

# Define the subfolders for original and resampled waveforms
wvtbls192 = os.path.join(wavetables_folder, '192k32b')  

# Define the output folder for resampled waveforms
pwrs2_folder = os.path.join(wavetables_folder, 'powersof2')  

# Create the pwrs2_folder if it doesn't exist
if not os.path.exists(pwrs2_folder):
    os.makedirs(pwrs2_folder)

# Calculate the nearest_192 higher power of 2
nearest_192_higher_power_of_2 = 2**np.ceil(np.log2(wavecycle_samples_target_192))

# Define the ratio of the nearest_192 higher power of 2 to wavecycle_samples_target_192
pwr_of_2_192_ratio = nearest_192_higher_power_of_2 / wavecycle_samples_target_192

print("Power of 2 192 ratio:", pwr_of_2_192_ratio)

# Calculate the target length for all files as the power of 2 target
pwr_of_2_192_target = int(round(wavecycle_samples_target_192 * pwr_of_2_192_ratio))
print(f"pwr_of_2_192_target: {pwr_of_2_192_target} samples")

# Calculate the new target length for all files
target_192_length = int(round(wavecycle_samples_target_192 * pwr_of_2_192_ratio))

# Define the subfolder within 'wavetables/powersof2' named '192' to save the interpolated segments
subfolder_192_name = "192"
powersof2_192_folder = os.path.join(wavetables_folder, "powersof2", subfolder_192_name)

# Create the powersof2_192_folder if it doesn't exist
if not os.path.exists(powersof2_192_folder):
    os.makedirs(powersof2_192_folder)

# Calculate the new target length for all 192k files
target_192_length = int(round(wavecycle_samples_target_192 * pwr_of_2_192_ratio))

for filename in os.listdir(wvtbls192):
    if filename.endswith('.wav'):  # Check if the file is a .wav file
        # Construct the full path for the input file using wvtbls192
        input_file_path = os.path.join(wvtbls192, filename)
        
        # Read the waveform and its sample rate
        data, original_sr = sf.read(input_file_path)
        
        # Use interpolate_seg to resample the segment to the new target length
        interpolated_data = interpolate_seg(data, original_sr, target_192_length)
        
        # Construct the full path for the output file using powersof2_192_folder
        pwr2_192_file_path = os.path.join(powersof2_192_folder, filename)
        
        # Save the interpolated segment to the powersof2_192_folder
        sf.write(pwr2_192_file_path, interpolated_data, original_sr, subtype='FLOAT')
        #print(f"Processed and saved: {filename} to {powersof2_192_folder}")

# block 9
# --- powers of two save as 48k24b---
print(f"\n48k24b")
wavecycle_samples_target_48 = round(48000 / sox_pitch)

# Define the subfolders for source waveforms
wvtbls48 = os.path.join(wavetables_folder, '192')  

# Define the output folder for resampled waveforms
pwrs2_folder = os.path.join(wavetables_folder, 'powersof2') 
nearest_48_higher_power_of_2 = 2**np.ceil(np.log2(wavecycle_samples_target_48))

# Create the pwrs2_folder if it doesn't exist
if not os.path.exists(pwrs2_folder):
    os.makedirs(pwrs2_folder)

# Define the subfolder within 'wavetables/powersof2' named '48' to save the interpolated segments
subfolder_48_name = "48"
powersof2_48_folder = os.path.join(wavetables_folder, "powersof2", subfolder_48_name)


print(f"Source waveforms folder set to: {wvtbls48}")
print(f"Resampled waveforms folder set to: {powersof2_48_folder}")

# Calculate the nearest_48 higher power of 2
nearest_48_higher_power_of_2 = 2**np.ceil(np.log2(wavecycle_samples_target_48))

# Define the ratio of the nearest_48 higher power of 2 to wavecycle_samples_target_48
pwr_of_2_48_ratio = nearest_48_higher_power_of_2 / wavecycle_samples_target_48


print("Power of 2, 48K ratio:", pwr_of_2_48_ratio)

# Calculate the target length for all files as the power of 2 target
pwr_of_2_48_target = int(round(wavecycle_samples_target_192 * pwr_of_2_48_ratio))


# Calculate the new target length for all files
target_48_length = int(round(wavecycle_samples_target_192 * pwr_of_2_48_ratio))


# Create the powersof2_48_folder if it doesn't exist
if not os.path.exists(powersof2_48_folder):
    os.makedirs(powersof2_48_folder)

# Calculate the new target length for all files
target_48_length = int(round(wavecycle_samples_target_192 * pwr_of_2_48_ratio))

for filename in os.listdir(wvtbls192):
    if filename.endswith('.wav'):  # Check if the file is a .wav file
        # Construct the full path for the input file using wvtbls192
        input_file_path = os.path.join(wvtbls192, filename)
        
        # Read the waveform and its sample rate
        data, original_sr = sf.read(input_file_path)
        
        # Use interpolate_seg to resample the segment to the new target length
        interpolated_data = interpolate_seg(data, original_sr, target_48_length)
        
        # Construct the full path for the output file using powersof2_48_folder
        output_pwr2_48_file_path = os.path.join(powersof2_48_folder, filename)
        
        # Save the interpolated segment to the powersof2_48_folder
        sf.write(output_pwr2_48_file_path, interpolated_data, original_sr, subtype='FLOAT')
        #print(f"Processed and saved: {filename} to {powersof2_48_folder}")

# Resample to 48k24b power of two files
# Define the source folder for the original waveforms (48k24b)
wvtbls48 = os.path.join(wavetables_folder, '48k24b')

# Define the original sample rate for the 48k files
original_sr_48 = 48000

# Define the output folder for resampled waveforms within 'powersof2'
pwr2_48_folder = os.path.join(wavetables_folder, 'powersof2', '48')

# Create the pwr2_48_folder if it doesn't exist
if not os.path.exists(pwr2_48_folder):
    os.makedirs(pwr2_48_folder)

# Calculate the nearest_48 higher power of 2
nearest_48_higher_power_of_2 = 2**np.ceil(np.log2(wavecycle_samples_target_48))

# Define the ratio of the nearest_48 higher power of 2 to wavecycle_samples_target_48
pwr_of_2_48_ratio = nearest_48_higher_power_of_2 / wavecycle_samples_target_48

# Calculate the new target length for all files
target_48_length = int(round(wavecycle_samples_target_192 * pwr_of_2_48_ratio))

# Counter to keep track of the file number
file_counter = 1

for filename in os.listdir(wvtbls48):
    if filename.endswith('.wav'):  # Check if the file is a .wav file
        # Construct the full path for the input file using wvtbls48
        input_file_path = os.path.join(wvtbls48, filename)
        
        # Read the waveform and its sample rate
        data, original_sr = sf.read(input_file_path)
        
        # Use interpolate_seg to resample the segment to the new target length
        interpolated_data = interpolate_seg(data, original_sr_48, target_48_length)
        
        # Construct the new filename using the desired format: {base}_pwr2_48k24b_NNNN{ext}
        pwr2_48_file_name = f"{base}_pwr2_48k24b_{file_counter:04d}{ext}"
        
        # Construct the full path for the output file using pwr2_48_folder
        output_pwr2_48_file_path = os.path.join(pwr2_48_folder, pwr2_48_file_name)
        
        # Save the interpolated segment to the pwr2_48_folder
        sf.write(output_pwr2_48_file_path, interpolated_data, original_sr_48, subtype='FLOAT')
        
        # Increment the file counter
        file_counter += 1

print(f"pwr_of_2_48_target: {pwr_of_2_48_target}")
print(f"pwr2_48_folder: {pwr2_48_folder}")
print(f"last saved: {output_pwr2_48_file_path}") 


# block 10
# --- Combining Specified Number of Wavecycles into a Single File ---
print("\n--- Combining Specified Number of Wavecycles ---")

# Define the actual source directory for 48k power of 2 waveforms
source_48k_pwr2 = os.path.join(pwrs2_folder, "48")  # This is the directory where the actual waveforms are stored

# Use the `/wavetables/192k32b` directory only to determine the count for total_files_in_48_folder
total_files_in_48_folder = len([f for f in os.listdir(wvtbls192) if f.endswith(ext)])  # Defined in uvar

# Specify the number of frames to combine
frames_to_combine = 256  # Adjust this value as needed

# Assuming other initializations and definitions are done above

# Calculate the highest possible starting frame
highest_start_frame = max(0, total_files_in_48_folder - frames_to_combine)

if highest_start_frame > 0:
    # Initialize an empty list to hold all 48k frames for combination
    combined_256_frames = []

    # Prompt the user for the starting frame within the valid range
    while True:
        start_frame_input = input(f"Enter the starting frame (1 to {highest_start_frame}): ")
        try:
            start_frame = int(start_frame_input)
            if 1 <= start_frame <= highest_start_frame:
                break
            else:
                print(f"Please enter a number within the range 1 to {highest_start_frame}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Continue with combining frames...
    # (Your existing code for combining frames goes here)

else:
    # Not enough frames to proceed with user input, use backfill method
    print("Not enough frames to complete forward fill. Performing forward-backward fill...")
    # Calculate the number of frames to fill backward
    backward_fill_count = frames_to_combine - total_files_in_48_folder

    # Initialize the list for the combined frames
    combined_frames = []

    # Forward fill with available frames
    for filename in sorted(os.listdir(source_48k_pwr2))[:total_files_in_48_folder]:
        if filename.endswith(ext):
            file_path = os.path.join(source_48k_pwr2, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)

    # Backward and inverted fill for the remainder
    for filename in sorted(os.listdir(source_48k_pwr2))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os.path.join(source_48k_pwr2, filename)
            waveform, sr = sf.read(file_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)

    # Concatenate all frames along the first axis (time axis)
    combined_frame = np.concatenate(combined_frames, axis=0)

    # Define the output file name
    output_file_name = f"{base}_combined_backfilled_{frames_to_combine}_frames{ext}"

    # Construct the full path for the output file
    output_file_path = os.path.join(pwrs2_folder, output_file_name)

    # Write the combined frames to the output file
    sf.write(output_file_path, combined_frame, sr, subtype='PCM_24')

    print(f"Backfilled combined file created at {output_file_path}")

# block 11
print("\nSerum 192k wt")

# Define the subfolders for original and resampled waveforms
wvtbls192 = os.path.join(wavetables_folder, '192k32b')
target_wt192_length = 2048  # Serum standard wt target length for all files

# Define and create the output folder for resampled waveforms
pwrs2_folder = os.path.join(wavetables_folder, 'powersof2')
os.makedirs(pwrs2_folder, exist_ok=True)

# Define and create the subfolder within 'powersof2' named 'wt192'
subfolder_wt192_name = "wt192"
powersof2_wt192_folder = os.path.join(pwrs2_folder, subfolder_wt192_name)
os.makedirs(powersof2_wt192_folder, exist_ok=True)

print(f"Source 192 waveforms folder set to: {wvtbls192}")
print(f"Resampled waveforms folder set to: {powersof2_wt192_folder}")

# Calculate the ratio of the 2048 to wavecycle_samples_target_192
pwr_of_2_wt192_ratio = target_wt192_length / wavecycle_samples_target_192
print(f"pwr_of_2_wt192_ratio: {pwr_of_2_wt192_ratio}")

# Calculate the target length for wt192K files as the power of 2 target
pwr_of_2_wt192_target = int(round(wavecycle_samples_target_192 * pwr_of_2_wt192_ratio))
print(f"2048 = {pwr_of_2_wt192_target}?")

for filename in os.listdir(wvtbls192):
    if filename.endswith('.wav'):
        wvtbl_192_path = os.path.join(wvtbls192, filename)
        data, original_sr = sf.read(wvtbl_192_path)
        interpolated_data = interpolate_seg(data, original_sr, target_wt192_length)
        output_wt192_file_path = os.path.join(powersof2_wt192_folder, filename)
        sf.write(output_wt192_file_path, interpolated_data, original_sr, subtype='FLOAT')

# Combining Specified Number of Wavecycles into a Single File
print("...combining 2048 sample by 256 frame serum wavetable as wav file...")

total_files_in_wt192_folder = len([f for f in os.listdir(wvtbls192) if f.endswith(ext)])
frames_to_combine_wt = 256  # 256 is the wt for serum standard
wt192_count_remainder = frames_to_combine_wt - total_files_in_wt192_folder
print(f"Available Frames = {total_files_in_wt192_folder}")
print(f"frames to fill (ignore if negative)  = {wt192_count_remainder}")

if total_files_in_wt192_folder == 0:
    print("Aborting: No frames available to combine. Please check the source directory.")
    exit()

elif total_files_in_wt192_folder == frames_to_combine_wt:
    print("2048 frames available. Proceeding with combination...")
    combined_wt192_frames = []
    
    for filename in sorted(os.listdir(powersof2_wt192_folder)):
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            combined_wt192_frames.append(waveform)

    combined_wt192_frame = np.concatenate(combined_wt192_frames, axis=0)
    combined_wt192_frame_out = f"{base}_wt192_full_frame{ext}"
    output_file_path = os.path.join(pwrs2_wt192_folder, combined_wt192_frame_out)
    sf.write(output_file_path, combined_wt192_frame, sr, subtype='FLOAT')
    print(f"Combined full frame saved as: {output_file_path}")

elif total_files_in_wt192_folder > frames_to_combine_wt:
    highest_start_frame_wt192 = max(1, total_files_in_wt192_folder - frames_to_combine_wt + 1)
    if highest_start_frame_wt192 > 1:
        # Prompt the user for the starting frame within the valid range
        while True:
            start_frame_wt192_input = input(f"Enter the starting frame (1 to {highest_start_frame_wt192}): ")
            try:
                start_frame_wt192 = int(start_frame_wt192_input)
                if 1 <= start_frame_wt192 <= highest_start_frame_wt192:
                    break
                else:
                    print(f"Please enter a number within the range 1 to {highest_start_frame_wt192}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        # Calculate the ending frame based on the selected starting frame
        ending_frame_wt192 = start_frame_wt192 + frames_to_combine_wt - 1

        # Initialize the list for the combined frames
        combined_192k_256_frames = []

        # Iterate through the files in the source directory starting from the selected frame
        for filename in sorted(os.listdir(powersof2_wt192_folder))[start_frame_wt192 - 1:ending_frame_wt192]:
            if filename.endswith(ext):
                # Construct the full path for the file
                file_path = os.path.join(powersof2_wt192_folder, filename)
                
                # Read the waveform data
                waveform, sr = sf.read(file_path)
                
                # Append the waveform data to the combined_192k_256_frames list
                combined_192k_256_frames.append(waveform)

        # Concatenate all frames along the first axis (time axis)
        combined_192k_2048_frame_wt192k = np.concatenate(combined_192k_256_frames, axis=0)

        # Define the output file name with the starting frame included
        combined_192k_2048_frame_out = f"{base}_wt192k24b_start{start_frame_wt192:04d}_{frames_to_combine_wt}_frame{ext}"

        # Construct the full path for the output file
        output_file_path = os.path.join(pwrs2_folder, combined_192k_2048_frame_out)

        # Write the combined frames to the output file
        sf.write(output_file_path, combined_192k_2048_frame_wt192k, sr, subtype='FLOAT')

        print(f"Combined {frames_to_combine_wt} frames starting from frame {start_frame_wt192} saved as: {output_file_path}")

else:
    # Not enough frames to complete forward fill. Performing forward-backward fill...
    print("Not enough frames to complete forward fill. Performing forward-backward fill...")
    backward_fill_count = frames_to_combine_wt - total_files_in_wt192_folder

    # Initialize the list for the combined frames
    combined_frames = []

    # Forward fill with available frames
    for filename in sorted(os.listdir(powersof2_wt192_folder))[:total_files_in_wt192_folder]:
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)

    # Backward and inverted fill for the remainder
    for filename in sorted(os.listdir(powersof2_wt192_folder))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os

    # Forward fill with available frames
    for filename in sorted(os.listdir(powersof2_wt192_folder))[:total_files_in_wt192_folder]:
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            combined_frames.append(waveform)
    
    # Backward and inverted fill for the remainder
    for filename in sorted(os.listdir(powersof2_wt192_folder))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_path = os.path.join(powersof2_wt192_folder, filename)
            waveform, sr = sf.read(file_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)
    
    # Concatenate all frames along the first axis (time axis)
    combined_frame = np.concatenate(combined_frames, axis=0)

    # Define the output file name
    output_file_name = f"{base}_combined_backfilled_{frames_to_combine_wt}_frames{ext}"

    # Construct the full path for the output file
    output_file_path = os.path.join(pwrs2_folder, output_file_name)

    # Write the combined frames to the output file
    sf.write(output_file_path, combined_frame, sr, subtype='FLOAT')

    print(f"Backfilled combined file created at {output_file_path}")
# block 12
print(f"\n\nDONE")
