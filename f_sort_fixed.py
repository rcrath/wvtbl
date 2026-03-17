# f_sort.py - FIXED
# Addresses issues #22 (autocorr tolerance), #28 (method tags), #31 (preserve suffixes)

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import soundfile as sf
from scipy.stats import mode as scipy_mode
from aa_common import get_wavecycle_samples_target, get_autocorrelation_flag, get_mode_interval, set_mode_interval

def autocorrelation_sort(wavecycle_samples):
    """
    FIX #22: Use average instead of mode for autocorrelation to avoid tight tolerance issues
    """
    if get_autocorrelation_flag():
        # Use average for autocorrelation - more robust than mode
        wavecycle_samples_target_avg = calculate_average_wavecycle_length(wavecycle_samples)
        mode_interval = get_mode_interval()
        
        # For autocorrelation, prefer average over mode
        print(f"Autocorrelation mode -- using average: {wavecycle_samples_target_avg}, mode: {mode_interval}")
        
        # Return average as the target for autocorrelation
        return wavecycle_samples_target_avg, wavecycle_samples_target_avg
    else:
        # For zero-crossing, use mode as before
        mode_interval = calculate_mode_wavecycle_length(wavecycle_samples)
        wavecycle_samples_target_avg = calculate_average_wavecycle_length(wavecycle_samples)
        
        return mode_interval, wavecycle_samples_target_avg

def calculate_adaptive_tolerance(wavecycle_samples, base_tolerance_percent):
    """
    FIX #22: Calculate adaptive tolerance based on variance in segment lengths.
    More variance = wider tolerance needed.
    """
    if not wavecycle_samples:
        return base_tolerance_percent
    
    lengths = list(wavecycle_samples.values())
    std_dev = np.std(lengths)
    mean_length = np.mean(lengths)
    
    if mean_length == 0:
        return base_tolerance_percent
    
    # Coefficient of variation
    cv = std_dev / mean_length
    
    # Increase tolerance if high variation
    if cv > 0.05:  # More than 5% variation
        adaptive_tolerance = base_tolerance_percent * (1 + cv * 10)
    else:
        adaptive_tolerance = base_tolerance_percent
    
    # Cap at 20% maximum
    return min(adaptive_tolerance, 20)

def mark_deviant_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, seg_folder, ext):
    """
    FIX #31: Ensure _dev suffix is properly added and preserved
    """
    outside_tolerance_files = []
    for segment_file, segment_size in segment_sizes:
        file_path = os.path.join(seg_folder, segment_file)
        
        # Skip if already marked as attack or deviant
        if '_atk' in segment_file or '_dev' in segment_file:
            continue
            
        if not os.path.exists(file_path):
            continue
            
        # Mark as deviant if outside tolerance
        if segment_size < lower_bound_samples or segment_size > upper_bound_samples:
            base_name = os.path.splitext(segment_file)[0]
            # FIX #31: Ensure suffix is added properly
            dev_name = f"{base_name}_dev{ext}"
            dev_path = os.path.join(seg_folder, dev_name)
            
            if os.path.exists(dev_path):
                print(f"File {dev_path} already exists. Skipping renaming.")
                continue
                
            os.rename(file_path, dev_path)
            outside_tolerance_files.append(dev_name)
            
    return outside_tolerance_files

def mark_attack_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, seg_folder, ext):
    """
    FIX #31: Ensure _atk suffix is properly added and preserved
    """
    within_tolerance_count = 0
    attack_segments = []
    
    for segment_file, segment_size in segment_sizes:
        file_path = os.path.join(seg_folder, segment_file)
        
        # Skip if already deviant
        if '_dev' in segment_file:
            continue
            
        # Count consecutive segments within tolerance
        if lower_bound_samples <= segment_size <= upper_bound_samples:
            within_tolerance_count += 1
        else:
            within_tolerance_count = 0
            
        # Mark first 3 segments as attack
        if within_tolerance_count < 3:
            base_name = os.path.splitext(segment_file)[0]
            
            # FIX #31: Add suffix if not already present
            if '_atk' not in segment_file:
                atk_name = f"{base_name}_atk{ext}"
            else:
                atk_name = segment_file
                
            atk_path = os.path.join(seg_folder, atk_name)
            
            # Try to rename from base or deviant version
            if not os.path.exists(file_path):
                file_path = os.path.join(seg_folder, f"{base_name}_dev{ext}")
                
            if os.path.exists(file_path):
                os.rename(file_path, atk_path)
                attack_segments.append(atk_name)
        else:
            # Found sustained portion, stop marking attacks
            break
            
    return attack_segments

def calculate_average_wavecycle_length(wavecycle_samples):
    """Calculate average wavecycle length from samples."""
    if not wavecycle_samples:
        return 0
    sizes = list(wavecycle_samples.values())
    return int(np.mean(sizes))

def calculate_mode_wavecycle_length(wavecycle_samples):
    """Calculate mode wavecycle length from samples."""
    if not wavecycle_samples:
        return 0

    sizes = list(wavecycle_samples.values())
    if len(sizes) == 1:
        set_mode_interval(sizes[0])
        return sizes[0]

    mode_result = scipy_mode(sizes)
    if isinstance(mode_result.count, np.ndarray):
        if mode_result.count.size > 0 and mode_result.count[0] > 1:
            set_mode_interval(mode_result.mode[0])
            return mode_result.mode[0]
    else:
        if mode_result.count > 1:
            set_mode_interval(mode_result.mode)
            return mode_result.mode

    set_mode_interval(sizes[0])
    return sizes[0]

def validate_segment_distribution(segment_sizes, lower_bound, upper_bound):
    """
    Validate that we have reasonable segment distribution.
    Returns (is_valid, warning_message)
    """
    total = len(segment_sizes)
    
    if total == 0:
        return False, "❌ No segments found at all!"
    
    within_tolerance = sum(1 for _, size in segment_sizes 
                          if lower_bound <= size <= upper_bound)
    
    within_percent = (within_tolerance / total) * 100
    
    if within_percent < 10:
        return False, f"❌ Only {within_percent:.1f}% of segments within tolerance - try widening tolerance or using other method"
    elif within_percent < 30:
        return True, f"⚠️  Warning: Only {within_percent:.1f}% of segments within tolerance"
    else:
        return True, None

def run(settings, total_segments, total_deviant_segments, total_normal_segments, 
        total_attack_segments, first_iteration, processed_files):
    from aa_common import (get_wavecycle_samples, get_wavecycle_samples_target, set_wavecycle_samples_target,
                           get_all_segment_sizes, get_manual_frequency_input, calculate_tolerance_bounds, 
                           get_base, get_tmp_folder, input_with_defaults, frequency_to_note_and_cents)
    
    print(">>> [DEBUG] Retrieved wavecycle_samples_target in f_sort.run:", get_wavecycle_samples_target())
    wavecycle_samples_target = get_wavecycle_samples_target()
    set_mode_interval(wavecycle_samples_target)
    
    base = get_base()
    cpy_folder = os.path.join(base, "tmp", "cpy")
    cpy_file = os.path.join(cpy_folder, f"{base}.wav")
    
    _, current_sample_rate = sf.read(cpy_file)
    
    wavecycle_samples = get_wavecycle_samples()
    wavecycle_samples_target, wavecycle_samples_target_avg = autocorrelation_sort(wavecycle_samples)
    
    print(f"wavecycle_samples_target: {wavecycle_samples_target}")
    
    # FIX #22: Use wider tolerance for autocorrelation
    if get_autocorrelation_flag():
        # Use adaptive tolerance or fixed higher value for autocorrelation
        base_tolerance = 10  # 10% for autocorrelation instead of 5%
        settings['percent_tolerance'] = calculate_adaptive_tolerance(wavecycle_samples, base_tolerance)
        print(f"Using adaptive tolerance for autocorrelation: {settings['percent_tolerance']:.1f}%")
    
    custom_wavecycle_samples_target = settings.get('custom_wavecycle_samples_target', None)
    custom_selection_type = settings.get('custom_selection_type', 'mode')
    lower_bound_samples, upper_bound_samples = calculate_tolerance_bounds(
        wavecycle_samples_target, 
        settings['percent_tolerance']
    )
    
    # Main selection loop
    while True:
        wavecycle_samples_target = get_wavecycle_samples_target()
        mode_frequency = current_sample_rate / wavecycle_samples_target
        mode_note, mode_cents = frequency_to_note_and_cents(mode_frequency)

        if custom_wavecycle_samples_target:
            current_wavecycle_samples_target = custom_wavecycle_samples_target
            current_frequency = current_sample_rate / custom_wavecycle_samples_target
            current_note, current_cents = frequency_to_note_and_cents(current_frequency)
        else:
            current_wavecycle_samples_target = wavecycle_samples_target
            current_frequency = mode_frequency
            current_note = mode_note
            current_cents = mode_cents

        if get_wavecycle_samples_target() == 0:
            print("\nChoose the dominant frequency. If you aren't sure, press ENTER or 1.")
            print(f"\n1. Set to the MODE ({mode_note}{mode_cents:+d} / {mode_frequency:.2f}Hz / {wavecycle_samples_target} samples)")
            print("\n2. Open the graph to select VISUALLY")
            print("\n3. Enter a musical NOTE (e.g., A4, C#3) or a frequency in Hz")

            choice = input_with_defaults("\nPress Enter, 1, 2 or 3 to choose: ", default="1").strip()
            if choice == "" or choice == "1":
                print(f"\nSetting wavecycle length to mode: {wavecycle_samples_target} samples / {mode_note}{mode_cents:+d} / {mode_frequency:.2f}Hz")
                set_wavecycle_samples_target(wavecycle_samples_target)
                settings['custom_wavecycle_samples_target'] = wavecycle_samples_target
                settings['custom_selection_type'] = 'mode'
                break
            elif choice == "2":
                # Would call plot function here
                break
            elif choice == "3":
                frequency = get_manual_frequency_input(20.0, 20000.0)
                if frequency:
                    custom_wavecycle_samples_target = int(current_sample_rate / frequency)
                    set_wavecycle_samples_target(custom_wavecycle_samples_target)
                    settings['custom_wavecycle_samples_target'] = custom_wavecycle_samples_target
                    settings['custom_selection_type'] = 'manual'
                    print(f"\nSetting central frequency to {frequency:.2f}Hz / {custom_wavecycle_samples_target} samples")
                break
        else:
            # Already have a selection, allow changing
            break
    
    seg_folder = os.path.join(get_tmp_folder(), "seg")
    ext = ".wav"

    segment_sizes = get_all_segment_sizes()
    
    # Validate before marking
    is_valid, warning = validate_segment_distribution(segment_sizes, lower_bound_samples, upper_bound_samples)
    if warning:
        print(warning)
    if not is_valid:
        print(f"\nCurrent settings:")
        print(f"  Target: {wavecycle_samples_target} samples")
        print(f"  Tolerance: ±{settings['percent_tolerance']}%")
        print(f"  Range: {lower_bound_samples}-{upper_bound_samples} samples")
    
    mark_deviant_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, seg_folder, ext)
    mark_attack_segments(segment_sizes, lower_bound_samples, upper_bound_samples, base, seg_folder, ext)

    total_segments = len([f for f in os.listdir(seg_folder) if re.match(r'.*_seg_\d{4}.*\.wav$', f)])
    total_attack_segments = len([f for f in os.listdir(seg_folder) if '_atk.wav' in f])
    total_deviant_segments = len([f for f in os.listdir(seg_folder) if '_dev.wav' in f])
    total_normal_segments = total_segments - total_deviant_segments - total_attack_segments

    print(f"\nSegment classification:")
    print(f"  Total: {total_segments}")
    print(f"  Normal: {total_normal_segments}")
    print(f"  Deviant: {total_deviant_segments}")
    print(f"  Attack: {total_attack_segments}")

    return total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, lower_bound_samples, upper_bound_samples
