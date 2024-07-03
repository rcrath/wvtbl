import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.widgets import Button
from aa_common import get_base

selected_segment = None
selected_sr = None  # Global variable to store the sample rate
connection_id = None  # Global variable to store the connection ID

def plot_wav_file_interactive(wav_file):
    global selected_sr, data, sr, connection_id
    if not os.path.exists(wav_file):
        print(f"File not found: {wav_file}")
        return

    data, sr = sf.read(wav_file)
    selected_sr = sr  # Store the sample rate globally
    time = np.linspace(0, len(data) / sr, num=len(data))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, data, label="Wav File")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wav File Visualization")
    ax.legend()

    # Add text to the plot
    ax.text(0.5, 0.9, 'Click on a starting point in the waveform', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    connection_id = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, data, sr, fig))
    plt.show()  # Use blocking show to keep the first plot open

def on_click(event, data, sr, fig):
    global selected_segment, connection_id
    if event.inaxes:
        x = event.xdata
        sample_index = int(x * sr)
        
        # Snap to the nearest multiple of 2048 samples
        sample_index = (sample_index // 2048) * 2048
        
        if sample_index < 0:  # If clicked to the left of the selector graph
            sample_index = 0  # Set start to the beginning of the file
        elif sample_index > len(data) - 256 * 2048:  # If clicked to the right beyond available samples
            sample_index = len(data) - 256 * 2048  # Adjust to the maximum valid start point
        
        selected_segment = highlight_selection(data, sr, sample_index)
        fig.canvas.mpl_disconnect(connection_id)  # Use stored connection ID to disconnect
        plt.close(fig)  # Close the initial plot

        # Open the selection plot
        plot_selection_graph(selected_segment, sr)

def plot_selection_graph(data, sr):
    global selected_segment  # Ensure selected_segment is accessible
    time = np.linspace(0, len(data) / sr, num=len(data))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, data, label="Selected Segment")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Selected Wavetable Segment")
    ax.legend()
    
    # Add text and buttons to the plot
    ax.text(0.5, 0.9, 'Is this ok?', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    ax_yes = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_no = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_yes = Button(ax_yes, 'Yes')
    btn_no = Button(ax_no, 'No')

    def on_yes(event):
        save_selection(selected_segment, selected_sr)
        plt.close('all')  # Close all plots

    def on_no(event):
        plt.close(fig)  # Close the selection plot
        plot_wav_file_interactive(wav_file)  # Reopen the initial plot

    btn_yes.on_clicked(on_yes)
    btn_no.on_clicked(on_no)
    
    plt.show(block=False)  # Non-blocking show for the second plot to allow interaction

    # Manually process events to keep the interaction responsive
    while plt.get_fignums():
        plt.pause(0.1)

def highlight_selection(data, sr, start_index):
    end_index = start_index + 256 * 2048
    selected_data = data[start_index:end_index]
    
    time = np.linspace(start_index / sr, end_index / sr, num=len(selected_data))
    
    return selected_data

def save_selection(data, sr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = get_base()
    base_folder = os.path.join(os.getcwd(), base)
    os.makedirs(base_folder, exist_ok=True)
    output_file = os.path.join(base_folder, f"{base}_selection_{timestamp}.wav")
    sf.write(output_file, data, sr)
    print(f"Saved selection to {output_file}")
