# h_interpolate.py - FIXED
# Addresses issue: Skip interpolation for segments already at correct length

import os
import shutil
import soundfile as sf
import aa_common

def run(total_segments, total_deviant_segments, total_normal_segments, total_attack_segments, settings):
    """
    Interpolate segments to 2048 samples.
    FIX: Skip files that are already exactly 2048 samples to avoid unnecessary resampling.
    """
    
    frames_folder = os.path.join(aa_common.get_tmp_folder(), "frames")
    seg_folder = os.path.join(aa_common.get_tmp_folder(), "seg")
    
    # Ensure frames folder exists
    os.makedirs(frames_folder, exist_ok=True)
    
    # Get discard choices from settings
    discard_atk = settings.get('discard_atk_choice', 'N').lower() == 'y'
    discard_dev = settings.get('discard_dev_choice', 'N').lower() == 'y'
    discard_good = settings.get('discard_good_choice', 'N').lower() == 'y'
    
    files_processed = 0
    files_skipped = 0
    files_copied = 0
    
    print("\nInterpolating segments to 2048 samples...")
    
    for filename in sorted(os.listdir(seg_folder)):
        if not filename.endswith('.wav'):
            continue
            
        # Check if file should be discarded based on settings
        if discard_atk and '_atk.wav' in filename:
            print(f"Skipping attack segment: {filename}")
            continue
        if discard_dev and '_dev.wav' in filename:
            print(f"Skipping deviant segment: {filename}")
            continue
        if discard_good and '_atk.wav' not in filename and '_dev.wav' not in filename:
            print(f"Skipping normal segment: {filename}")
            continue
        
        input_path = os.path.join(seg_folder, filename)
        output_path = os.path.join(frames_folder, filename)
        
        try:
            # Load the audio file
            data, sr = sf.read(input_path)
            current_length = len(data)
            
            # FIX: Check if already correct length
            if current_length == aa_common.wavecycle_size:
                # Already correct length - just copy without resampling
                print(f"✓ Copying {filename} (already {aa_common.wavecycle_size} samples)")
                shutil.copy2(input_path, output_path)
                files_copied += 1
            else:
                # Needs interpolation
                print(f"↻ Interpolating {filename}: {current_length} → {aa_common.wavecycle_size} samples")
                interpolated = aa_common.interpolate_seg(data, sr)
                sf.write(output_path, interpolated, sr, subtype='FLOAT')
                files_processed += 1
                
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            files_skipped += 1
    
    print(f"\nInterpolation complete:")
    print(f"  Files interpolated: {files_processed}")
    print(f"  Files copied (already correct): {files_copied}")
    print(f"  Files skipped/errors: {files_skipped}")
    
    return frames_folder
