# d_pitch.py

import os
from scipy import stats
import aa_common
import numpy as np
import sys

def run():
    # print("running d_pitch")
    # Constants
    lowest_freq = 20
    highest_freq = 880
    confidence_threshold = 0.50
    freq_est_manually_set = False

    # Retrieve the necessary paths from aa_common
    base_prep_192k32b_path = os.path.join(aa_common.get_tmp_folder(), f"{aa_common.get_base()}_prep_192k32b.wav")

    # Pitch finding using crepe neural net
    print("\n\n====== BEHOLD AS THE MACHINE LEARNS! (I.E, ignore the following.) ==========\n\n")
    frequency_test, confidence_test = aa_common.test_crepe(base_prep_192k32b_path)
    print("⌃⌃⌃⌃⌃⌃⌃⌃⌃⌃END OF JUNK TO IGNORE⌃⌃⌃⌃⌃⌃⌃⌃⌃⌃\n\n\n")

    # Define the tolerance range
    lower_bound_crepe = lowest_freq
    upper_bound_crepe = highest_freq

    # Prepare frequencies for mode calculation
    filtered_frequencies = [frequency for frequency in frequency_test if lower_bound_crepe <= frequency <= upper_bound_crepe]
    mapped_frequencies = [aa_common.frequency_to_note(f) for f in filtered_frequencies]

    if len(mapped_frequencies) > 0:
        mode_result = stats.mode(mapped_frequencies)
        if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
            mode_frequency = mode_result.mode[0]
        else:
            mode_frequency = mode_result.mode

        mode_confidences = [confidence_test[i] for i, f in enumerate(filtered_frequencies) if aa_common.frequency_to_note(f) == mode_frequency]
        mode_confidence_avg = np.mean(mode_confidences) if len(mode_confidences) > 0 else 0
        freq_est = mode_frequency if mode_frequency else 0  # Use detected frequency or a default value
        note_est_cents = aa_common.frequency_to_note_and_cents(freq_est)
        mode_confidence_avg_prcnt = round(mode_confidence_avg * 100)
        print(f"Detected frequency: {round(freq_est)}Hz or {note_est_cents} with {mode_confidence_avg_prcnt}% confidence.")
    else:
        print("No frequencies to process.")
        mode_frequency = None
        mode_confidence_avg = 0

    # Assume mode_frequency and mode_confidence_avg have been determined as before
    if freq_est_manually_set:
        print(f"Detected mode frequency: {mode_frequency} Hz with confidence {round(mode_confidence_avg * 100)}%.")
        choice_prompt = f"Choose frequency source: \n1. Detected mode ({mode_frequency} Hz), \n2. Manually entered ({freq_est} Hz) \n[default: 1]: "
        user_choice = input(choice_prompt).strip()

        if user_choice == '2':
            mode_frequency = freq_est
            print(f"Using manually entered frequency: {freq_est} Hz.")
        else:
            print(f"Proceeding with detected mode frequency: {mode_frequency} Hz.")
    elif mode_confidence_avg < confidence_threshold:
        print("Pitch detection results are not reliable.")
        freq_est = aa_common.get_manual_frequency_input(lowest_freq, highest_freq)
        if freq_est is not None:
            mode_frequency = freq_est
            freq_est_manually_set = True
            print(f"Using manually entered frequency: {freq_est} Hz.")
        else:
            print("No manual frequency input provided. Unable to proceed.")
    else:
        # print(f"Proceeding with high-confidence mode frequency: {mode_frequency} Hz.")
        pass

    wavecycle_samples_target_192 = round(192000 / mode_frequency) if mode_frequency else None

    if not wavecycle_samples_target_192 or wavecycle_samples_target_192 <= 0:
        print("Unable to proceed without a valid target wave cycle sample count.")
        sys.exit(1)

    # print("\nFinal frequency decision:")
    if mode_frequency is not None:
        # print(f"Final Mode Frequency: {mode_frequency} Hz.")
        pass
    else:
        print("No mode frequency determined; unable to proceed.")
    
    return freq_est

if __name__ == "__main__":
    run()
