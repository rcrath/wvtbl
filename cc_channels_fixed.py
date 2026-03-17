import os
import aa_common
import numpy as np
import soundfile as sf
import resampy
from scipy.signal import butter, filtfilt


def load_audio(file_path):
    """Function to load audio data from a file."""
    data, sample_rate = sf.read(file_path)
    return data, sample_rate

def save_audio(file_path, data, sample_rate):
    """
    Save the audio data to a specified file path with the given sample rate.
    """
    sf.write(file_path, data, sample_rate, subtype='FLOAT')

def cleanup_files(file_paths, selected_files):
    """
    Delete files that were not selected for further processing.
    """
    for name, path in selected_files.items():
        print(f"Proceed with {path}")

def extract_channels(data):
    """
    Extract channels from audio data (already loaded).
    """
    # Check if the data is stereo
    if data.ndim == 2 and data.shape[1] == 2:  # Stereo file
        # Extract Left and Right channels
        left = data[:, 0]
        right = data[:, 1]
        mid = (left + right) / 2
        side = (left - right) / 2

        print(f"Stereo detected: Left channel length: {len(left)}, Right channel length: {len(right)}")
        print(f"Mid channel length: {len(mid)}, Side channel length: {len(side)}")

        return {
            'Left': left,
            'Right': right,
            'Mid': mid,
            'Side': side
        }
    elif data.ndim == 1:  # Mono file
        print(f"Mono detected: Channel length: {len(data)}")
        return {'Mono': data}
    else:
        raise ValueError("Unsupported audio format: Expected 1D (mono) or 2D (stereo) data.")


def save_channel_files(channels, sample_rate, base, tmp_folder):
    file_paths = {}
    for name, channel_data in channels.items():
        filtered_channel_data = high_pass_filter(channel_data, sample_rate)
        file_name = f"{base}_{name}.wav"
        file_path = os.path.join(tmp_folder, file_name)
        sf.write(file_path, filtered_channel_data, sample_rate, subtype='FLOAT')
        file_paths[name] = file_path
    return file_paths


def parse_channel_selection(selections, total_channels):
    """
    Parse channel selection input supporting both ranges and comma-separated values.
    Fixes issue #32 - handles mixed input like "1-2,4" properly.
    
    Parameters:
    - selections: str, user input like "1", "1,3", "1-2", or "1-2,4"
    - total_channels: int, total number of available channels
    
    Returns:
    - list of selected indices (0-based)
    """
    selected_indices = set()
    
    try:
        # Split by comma first
        parts = selections.split(',')
        
        for part in parts:
            part = part.strip()
            
            if '-' in part:
                # Handle range (e.g., "1-2")
                start, end = map(int, part.split('-'))
                # Convert to 0-based indexing and add range
                selected_indices.update(range(start - 1, end))
            else:
                # Handle single number (e.g., "4")
                idx = int(part)
                # Convert to 0-based indexing
                selected_indices.add(idx - 1)
        
        # Filter out invalid indices
        selected_indices = {i for i in selected_indices if 0 <= i < total_channels}
        
        if not selected_indices:
            return None
            
        return sorted(list(selected_indices))
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing selection: {e}")
        print("Please use format like: 1 or 1,3 or 1-2 or 1-2,4")
        return None


def choose_channels(file_paths):
    """
    Select channels to proceed with. Supports defaults for 'Mid' or 'Mono' channels.
    FIXED: Issue #32 - now properly handles mixed range/comma input like "1-2,4"
    """
    # Check if the file is mono
    if 'Mono' in file_paths:
        print(f"Proceed with Mono channel: {file_paths['Mono']}")
        return {'Mono': file_paths['Mono']}
    else:
        # Reorder channels for stereo files: "Mid", "Side", "Left", "Right"
        ordered_channels = []
        if 'Mid' in file_paths:
            ordered_channels.append('Mid')
        if 'Side' in file_paths:
            ordered_channels.append('Side')
        if 'Left' in file_paths:
            ordered_channels.append('Left')
        if 'Right' in file_paths:
            ordered_channels.append('Right')

        print("\nAvailable channels:")
        for i, name in enumerate(ordered_channels, start=1):
            print(f"{i}. {name} ({os.path.basename(file_paths[name])})")
        
        # Use input_with_defaults to handle default selection
        while True:
            selections = aa_common.input_with_defaults(
                "\nEnter the numbers of the channels to proceed with\n"
                "(e.g., 1 for Mid only, 1,3 for Mid and Left, 1-2 for Mid and Side, or 1-2,4 for Mid, Side, and Right)\n"
                "Default=1 (Mid): ",
                default="1"
            ).strip()

            # Parse the input using the new robust parser
            selected_indices = parse_channel_selection(selections, len(ordered_channels))
            
            if selected_indices is not None:
                selected_channels = [ordered_channels[i] for i in selected_indices]
                print(f"Selected channels: {', '.join(selected_channels)}")
                return {name: file_paths[name] for name in selected_channels}
            else:
                print("Invalid selection. Please try again.")


def high_pass_filter(data, sample_rate, cutoff_freq=10):
    """
    Apply a Butterworth high-pass filter to remove DC offset.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data).astype(np.float32)

def run():
    tmp_folder = aa_common.get_tmp_folder()
    base = aa_common.get_base()

    cpy_folder = os.path.join(tmp_folder, "cpy")
    if not os.path.exists(cpy_folder):
        os.makedirs(cpy_folder)
        
    cpy_file = os.path.join(tmp_folder, "cpy", f"{base}.wav")
    
    # Load the waveform and sample rate from the input file
    cpy_file_data, sample_rate = sf.read(cpy_file)
    
    # Extract channels
    channels = extract_channels(cpy_file_data)

    channel_files = save_channel_files(channels, sample_rate, base, cpy_folder)

    # Choose channels to proceed with
    selected_channels = choose_channels(channel_files)

    # List to store the paths of processed files
    processed_files = []

    # Process each selected channel
    for name, channel_path in selected_channels.items():
        # Load the waveform and sample rate from the selected channel file
        channel_data, sample_rate = load_audio(channel_path)

        # Save the audio data to the cpy folder without the suffix
        channel_file_path = os.path.join(cpy_folder, f"{base}_{name}.wav")
        save_audio(channel_file_path, channel_data, 192000)
        processed_files.append(channel_file_path)

    # Cleanup files that were not chosen or processed
    cleanup_files(channel_files, {name: path for name, path in selected_channels.items()})

    # Return the list of processed files
    return processed_files
