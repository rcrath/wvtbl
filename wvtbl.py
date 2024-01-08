
# block 1 
import os
import sys
import wave
import crepe
import aubio
import librosa
import re
import shutil
import resampy
import subprocess
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.interpolate import interp1d
import warnings

# Check if the correct number of command-line arguments are provided
if len(sys.argv) >= 4:
    start_file = sys.argv[1]  # The name of the start file, e.g., 'gtrsaw07a_233.wav'
    base = sys.argv[2]  # The base name for directory structure, e.g., 'gtrsaw07a1_233hz'
    freq_est = float(sys.argv[3])  # The frequency estimate
else:
    print("Usage: python wvtbl.py <start_file>.wav <base> <freq_est>")
    sys.exit(1)

# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)

# create base folder
os.makedirs(base, exist_ok=True)  # 'base' is a provided variable from the command line
print(f"base: {base}")
# Set the file extension
ext = ".wav"

# Define the source folder and construct the full path to the start file
source_folder = "source"
print(f"source_folder: {source_folder}")
start_file= os.path.join(source_folder, start_file)  # Using the first argument for the start file name

# Check if the start file exists within the source folder
if not os.path.exists(start_file):
    print(f"The start file '{start_file}' does not exist in the source folder. Please ensure the file is there and try again.")
    sys.exit(1)

# define tmpfile for upsampled full wavfile
base_prep_192k32b = f"{base}-prep_192k32{ext}"

# Load the waveform and sample rate from the input file
sample_rate, start_file_data = wavfile.read(start_file)

# Create the "tmp" folder if it doesn't exist
tmp_folder = "tmp"
create_folder(tmp_folder)

# Filter out the specific warning...this is not working.  
# warnings.filterwarnings("ignore", message="Chunk (non-data) not understood, skipping it.")


# Declare amplitude_tolerance_db as a global variable
amplitude_tolerance_db = -60

# Define specific subdirectories inside the base
single_cycles_folder = os.path.join(base, 'single_cycles')
os.makedirs(single_cycles_folder, exist_ok=True)

single_cycles_192k32b = os.path.join(single_cycles_folder, '192k32b')
os.makedirs(single_cycles_192k32b, exist_ok=True)

single_cycles_192k_2048samples = os.path.join(single_cycles_folder, '192k_2048samples')
os.makedirs(single_cycles_192k_2048samples, exist_ok=True)

# Define and create the output folder for the 256 frame combined files for Serum wavetables
serum_wavetable_folder = os.path.join(base, 'serum_wavetable')
os.makedirs(serum_wavetable_folder, exist_ok=True)

serum_2048x256 = os.path.join(serum_wavetable_folder, 'serum_2048x256')
os.makedirs(serum_2048x256, exist_ok=True)

base_prep_192k32b_path = os.path.join(tmp_folder, base_prep_192k32b)




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

# Function to run SoX and get pitch information
def get_pitch_using_sox(base_prep_192k32b, tmp_folder="tmp"):
    try:
        # Construct the path to the prep file
        base_prep_path_file = os.path.join(tmp_folder, base_prep_192k32b)

        # Run SoX on the prep file and extract the frequency estimate
        command = f"sox {base_prep_path_file} -n stat 2>&1 | grep 'frequency' | tr -cd '0-9.'"
        result = os.popen(command).read()

        return float(result) if result else 0
    except Exception as e:
        print(f"Error in get_pitch_using_sox: {e}")
 
# Pitch finding using neural net        
def test_crepe(base_prep_192k32b):
    sr, audio = wavfile.read(start_file)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    return frequency, confidence  # Returns frequency and confidence of prediction

def run_aubio_pitch(base_prep_192k32b_data_path, tolerance=0.8, method='default'):
    audio, sr = sf.read(base_prep_192k32b_data_path)  # Reads file and converts to 'float32'
    audio = audio.astype(np.float32)  # Cast audio to float32
    pitch_o = aubio.pitch(method, 2048, 512, sr)
    pitch_o.set_tolerance(tolerance)
    samples = np.array([pitch_o(samples.astype(np.float32))[0] for samples in librosa.frames_to_samples(librosa.util.frame(audio, frame_length=512, hop_length=512))])
    return samples

def run_librosa_piptrack(base_prep_192k32b, sr):
    pitches, magnitudes = librosa.piptrack(y=base_prep_192k32b, sr=sr)
    return pitches


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

# find pitch crepe
def run_crepe(audio, sr, model_capacity='tiny'):
    time, frequency, confidence, activation = crepe.predict(audio, sr, model_capacity=model_capacity)
    return time, frequency, confidence

def combine_and_save_frames(start_frame, frame_count):
    combined_frames = []
    ending_frame = start_frame + frame_count - 1

    for filename in sorted(os.listdir(single_cycles_192k_2048samples))[start_frame - 1:ending_frame]:
        if filename.endswith(ext):
            file_2048_path = os.path.join(single_cycles_192k_2048samples, filename)
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

    for filename in sorted(os.listdir(single_cycles_192k_2048samples))[:total_files_in_single_cycles_192k32b]:
        if filename.endswith(ext):
            file_2048_path = os.path.join(pwr2_192_folder, filename)
            waveform, sr = sf.read(file_2048_path)
            combined_frames.append(waveform)
    
    for filename in sorted(os.listdir(pwr2_192_folder))[-backward_fill_count:]:
        if filename.endswith(ext):
            file_2048_path = os.path.join(pwr2_192_folder, filename)
            waveform, sr = sf.read(file_2048_path)
            inverted_waveform = -np.flip(waveform)
            combined_frames.append(inverted_waveform)
    
    combined_frame = np.concatenate(combined_frames, axis=0)
    backfill_file_name = f"{base}_combined_backfilled_{frames_to_combine_wt}_frames{ext}"
    backfill_file_path = os.path.join(single_cycles_192k_2048samples, backfill_file_name)
    sf.write(backfill_file_path, combined_frame, sr, subtype='FLOAT')
    print(f"Backfilled combined file created at {backfill_file_path}")


# block 3

# begin upsample section
print("\nUpsample section")

# Load the waveform and sample rate from the input file
sample_rate, bit_depth = wavfile.read(start_file)


y, sr = librosa.load(start_file, sr=None)  # Load the file with its original sample rate
# print(f"DEBUG: base_prep_path_file: {base_prep_path_file}")


# Calculate the duration of the input waveform in seconds
duration_seconds = len(start_file) / sample_rate

# Calculate the number of samples needed to interpolate to 192k while keeping the same duration
target_samples_192k = round(192000 * duration_seconds)

# Resample the input waveform to 192k samples using the best interpolation method
interpolated_input_192k32b = interpolate_best(start_file_data, sample_rate, 192000)
print(f"base_prep_192k32b: {base_prep_192k32b}")
print(f"interpolated_input_192k32b: {interpolated_input_192k32b}")


# Save the interpolated input as a temporary wave file
wavfile.write(os.path.join(tmp_folder, base_prep_192k32b), 192000, interpolated_input_192k32b)

# set variables for reading files
base_prep_192k32b_data, sr = librosa.load(os.path.join(tmp_folder, base_prep_192k32b), sr=None)
base_prep_192k32b_data_path = os.path.join(tmp_folder, base_prep_192k32b)
print(f"base_prep_192k32b_data: {base_prep_192k32b_data}")
print(f"base_prep_192k32b_data_path: {base_prep_192k32b_data_path}")

print(f"path: {tmp_folder}/{base_prep_192k32b}")
print(f"base_prep_192k32b_data: {base_prep_192k32b_data}")

# Add a 5-millisecond fade in and fade out
fade_samples = int(0.001 * 192000)  # 1 milliseconds at 192k samples/second
fade_window = np.linspace(0, 1, fade_samples)

interpolated_input_192k32b = interpolated_input_192k32b.astype(np.float32)
interpolated_input_192k32b[:fade_samples] *= fade_window
interpolated_input_192k32b[-fade_samples:] *= fade_window[::-1]


# begin Pitch detection
print("\nPitch detection")
# Continue with pitch detection using different functions
# Initialize wavecycle_samples_target_192 with a default or null value
wavecycle_samples_target_192 = None  # Or some default value if appropriate

# Calculate pitch using SoX
try:
    sox_pitch = get_pitch_using_sox(base_prep_192k32b)
    print(f"base_prep_192k32b: {base_prep_192k32b}")

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


sox_pitch = get_pitch_using_sox(base_prep_192k32b)
print(f"SoX: {sox_pitch}")
librosa_pitch = run_librosa_piptrack(base_prep_192k32b_data, sr)
print(f"Librosa: {librosa_pitch}")
aubio_pitch_small = run_aubio_pitch(base_prep_192k32b_data_path, sr, method='yin')
aubio_pitch_mid = run_aubio_pitch(base_prep_192k32b_data_path, sr, method='fcomb')
aubio_pitch_large = run_aubio_pitch(base_prep_192k32b_data_path, sr, method='yinfft')
print(f"Aubio (Small): {aubio_pitch_small}")
print(f"Aubio (Mid): {aubio_pitch_mid}")
print(f"Aubio (Large): {aubio_pitch_large}")
Apply smoothing
smoothed_frequency_small = smooth(frequency_small, window_len=5)  # Change window_len for different levels of smoothing
smoothed_frequency_mid = smooth(frequency_mid, window_len=11)
smoothed_frequency_large = smooth(frequency_large, window_len=21)
# Test with different model capacities
time_small, frequency_small, confidence_small = run_crepe(y, sr, 'tiny')
time_mid, frequency_mid, confidence_mid = run_crepe(y, sr, 'medium')
time_large, frequency_large, confidence_large = run_crepe(y, sr, 'full')


print(f"CREPE (Small): {smoothed_frequency_small}")
print(f"CREPE (Mid): {smoothed_frequency_mid}")
print(f"CREPE (Large): {smoothed_frequency_large}")

print(f"Aubio (Small): {aubio_pitch_small}")
print(f"Aubio (Mid): {aubio_pitch_mid}")
print(f"Aubio (Large): {aubio_pitch_large}")


# --- "upsample" section ends here ---

print("\n --- PAUSE ---\n")
'''

'''