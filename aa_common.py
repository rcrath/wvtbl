
#aa_common
import os
import sys
import shutil
import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf
from scipy import stats
import crepe
import resampy

# Global variables
source_folder = "source"
ext = ".wav"
_start_file_name = ""
_start_file = ""
_base = ""
tmp_folder = ""
global_settings = {}  # Global settings dictionary

def initialize_settings():
    global global_settings
    # Initial settings with default values
    global_settings = {
        'freq_est': 'enter',  # Default action is to proceed without setting
        'percent_tolerance': 5,  # Default tolerance percent
        'discard_atk_choice': 'N',  # Default choice for discarding attack segments
        'discard_dev_choice': 'N',  # Default choice for discarding deviant segments
        'discard_good_choice': 'N',  # Default choice for discarding good segments
        'cleanup_choice': 'Y',  # Default choice for cleanup
        'accept_current_settings': False  # Track if settings are accepted
    }
    return global_settings

def update_settings(settings):
    global global_settings
    global_settings.update(settings)
    return global_settings

def prompt_update_settings(freq_est, settings):
    lowest_freq = 20  # Define lowest and highest frequency bounds
    highest_freq = 880
    
    # Check if settings were already accepted
    if settings.get('accept_current_settings', False):
        print("     Proceeding with current settings.\n\n\n")
        return settings

    accept_defaults = input("\n\n    Accept current settings? (Y/n, default=Y): ").strip().upper() or 'Y'
    if accept_defaults == 'Y':
        print("     Proceeding with current settings.\n\n\n")
        settings['accept_current_settings'] = True
        return settings

    # Update settings only if not accepting defaults
    settings['freq_note_input'] = input(f"Enter the frequency in Hz (between {lowest_freq}Hz and {highest_freq}Hz), \nOr note (no flats) with octave \n(e.g., A3, A#3, B3, C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4), \nCurrent: {freq_est}, or press <enter> to proceed without setting it.\nHz, Note, or <enter>: ").strip().upper() or 'enter'
    
    # Ensure the input is valid
    while True:
        freq_input = settings['freq_note_input']
        if freq_input == 'enter':
            break
        if freq_input.replace('.', '', 1).isdigit() and lowest_freq <= float(freq_input) <= highest_freq:
            freq_est = float(freq_input)
            print(f"Setting frequency to {freq_est}Hz ({frequency_to_note_and_cents(freq_est)[0]} and {frequency_to_note_and_cents(freq_est)[1]} cents)")
            break
        else:
            new_freq_est = note_to_frequency(freq_input)
            if new_freq_est and lowest_freq <= new_freq_est <= highest_freq:
                freq_est = new_freq_est
                print(f"Setting frequency to {freq_est}Hz ({frequency_to_note_and_cents(freq_est)[0]} and {frequency_to_note_and_cents(freq_est)[1]} cents)")
                break
            elif freq_input == 'Q':
                print("Quitting script.")
                sys.exit()
            else:
                print("Invalid frequency or note+octave. Please enter again.")
                settings['freq_note_input'] = input(f"Enter the frequency in Hz (between {lowest_freq}Hz and {highest_freq}Hz), \nOr note (no flats) with octave \n(e.g., A3, A#3, B3, C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4), \nCurrent: {freq_est}, or press <enter> to proceed without setting it.\nHz, Note, or <enter>: ").strip().upper() or 'enter'
    
    settings['freq_note_input'] = str(freq_est)

    # Error-checking for percent_tolerance input
    while True:
        percent_input = input(f"Set deviation tolerance from target length (default={settings['percent_tolerance']}%): ").strip()
        if percent_input == '':
            break
        try:
            settings['percent_tolerance'] = float(percent_input)
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    settings['discard_atk_choice'] = input("Discard attack segments? (y/N, default=N): ").strip().upper() or 'N'
    settings['discard_dev_choice'] = input("Discard deviant segments? (y/N, default=N): ").strip().upper() or 'N'
    settings['discard_good_choice'] = input("Discard good segments? (y/N, default=N): ").strip().upper() or 'N'
    settings['cleanup_choice'] = input("Perform cleanup? (Y/n, default=Y): ").strip().upper() or 'Y'
    settings['accept_current_settings'] = False
    return settings

def cleanup_tmp_folder(tmp_folder, prep_file_name):
    for file in os.listdir(tmp_folder):
        file_path = os.path.join(tmp_folder, file)
        if file != prep_file_name:
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error: {e}. Skipping file: {file_path}")


def list_and_select_wav_files(source_folder):
    print("\n\nSource Files:")
    files = [f for f in os.listdir(source_folder) if f.endswith(ext)]
    files.sort(key=lambda x: x.lower())
    for i, file in enumerate(files):
        print(f"{i+1}: {file}")
    print("\nEnter the number of the file to select, or type 'q' to exit.")
    selection = input("Selection: ").strip()
    if selection.lower() == 'q':
        print("Quitting script.")
        sys.exit()
    try:
        selected_index = int(selection) - 1
        if 0 <= selected_index < len(files):
            return files[selected_index]
        else:
            print("Invalid selection. Please try again.")
            return list_and_select_wav_files(source_folder)
    except ValueError:
        print("Please enter a valid number or 'q'.")
        return list_and_select_wav_files(source_folder)

def decide_start_file():
    global _start_file_name, _start_file, _base, tmp_folder
    if len(sys.argv) >= 2:
        _start_file_name = sys.argv[1]
    else:
        _start_file_name = list_and_select_wav_files(source_folder)
    _start_file = os.path.join(source_folder, _start_file_name)
    if not os.path.exists(_start_file):
        print(f"'{_start_file}' does not exist. Please check the file name and try again.")
        sys.exit(1)
    _base = os.path.splitext(os.path.basename(_start_file))[0]
    # Set the tmp_folder to be inside the _base folder in the top directory
    tmp_folder = os.path.join(_base, "tmp")
    print(f"\nProcessing file: {_start_file}\n")

def get_start_file_name():
    return _start_file_name

def get_start_file():
    return _start_file

def get_base():
    return _base

def get_tmp_folder():
    return tmp_folder

def ensure_tmp_folder():
    if not os.path.exists(_base):
        os.makedirs(_base)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    if os.path.exists(tmp_folder):
        pass
    else:
        print(f"Failed to create temporary folder: {tmp_folder}")

def test_crepe(base_prep_192k32b_path):
    sr, audio = wavfile.read(base_prep_192k32b_path)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True) 
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

def note_to_frequency(note):
    """
    Convert a musical note to its corresponding frequency.
    Assumes A4 = 440Hz as standard tuning.

    Parameters:
    - note: str, the musical note in the format 'NoteOctave' (e.g., 'A4', 'C#3').

    Returns:
    - float: The frequency corresponding to the note, or None if input is invalid.
    """
    A4 = 440
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Validate input
    if len(note) < 2 or not note[-1].isdigit():
        print("Invalid note format. Please enter a note followed by its octave (e.g., A4, C#3).")
        return None

    octave = int(note[-1])  # Extract the octave number
    note_name = note[:-1]  # Extract the note name (without octave)
    
    if note_name in notes:
        # Calculate the note's index in the octave from C0 up to the note
        note_index = notes.index(note_name) - notes.index('A') + (octave - 4) * 12
        # Calculate the frequency
        return A4 * (2 ** (note_index / 12))
    else:
        print("Invalid note name. Please enter a valid musical note (e.g., A4, C#3).")
        return None

def frequency_to_note_and_cents(frequency, A4=440):
    """
    Convert frequency to the nearest note and the cents offset from that note.

    Parameters:
    - frequency: float, the frequency to convert.
    - A4: float, the frequency of A4 (default is 440 Hz).

    Returns:
    - str: Note name and octave (e.g., 'A4').
    - int: Cents offset from the nearest note.
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    c0 = A4 * pow(2, -4.75)
    half_steps_above_c0 = round(12 * np.log2(frequency / c0))
    note = notes[half_steps_above_c0 % 12]
    octave = half_steps_above_c0 // 12
    exact_frequency = c0 * pow(2, half_steps_above_c0 / 12)
    cents = round(1200 * np.log2(frequency / exact_frequency))
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
def get_segment_sizes(base, tmp_folder, ext):
    segment_files = [f for f in os.listdir(tmp_folder) if f.startswith(f"{base}_seg") and f.endswith(ext)]
    segment_sizes = []
    for segment_file in segment_files:
        file_path = os.path.join(tmp_folder, segment_file)
        data, _ = sf.read(file_path)
        segment_sizes.append((segment_file, len(data)))
    return segment_sizes

def recount_segments(base, tmp_folder, ext):
    segment_sizes = get_segment_sizes(base, tmp_folder, ext)
    total_segments = len(segment_sizes)
    total_deviant_segments = len([f for f in segment_sizes if '_dev' in f[0]])
    total_normal_segments = len([f for f in segment_sizes if '_seg' in f[0] and not any(suffix in f[0] for suffix in ['_atk', '_dev'])])
    total_attack_segments = len([f for f in segment_sizes if '_atk' in f[0]])
    return total_segments, total_deviant_segments, total_normal_segments, total_attack_segments

# aa_common.py

def print_segment_info(total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, freq_est, wavecycle_samples_target_192, settings, lower_bound_samples, upper_bound_samples, discarding_atk, discarding_dev, discarding_good):
    # Print the segment information
    print(f"\nSEGMENT INFORMATION: \nTarget samples per wavecycle: {wavecycle_samples_target_192} at frequency {round(freq_est)}Hz.")
    print(f"Valid segment range within +/-{settings['percent_tolerance']}% tolerance: {round(lower_bound_samples)} to {round(upper_bound_samples)} samples.")

    # Define the descriptions and corresponding variables
    descriptions = [
        "Total number of wavetable segments",
        "Number of segments outside the sample tolerance range",
        "Number of segments within the sample tolerance range",
        "Number of attack segments"
    ]

    values = [total_segments, total_deviant_segments, total_normal_segments, total_attack_segments]

    # Print the descriptions with leading dots and right-justified integers
    for description, value in zip(descriptions, values):
        print(f"    {description:.<55}{value:4d}")

def interpolate_seg(data, original_sr, target_samples):
    current_samples = len(data)
    target_sr = (target_samples / current_samples) * original_sr
    resampled_data = resampy.resample(data, sr_orig=original_sr, sr_new=target_sr, axis=0)
    return resampled_data

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

