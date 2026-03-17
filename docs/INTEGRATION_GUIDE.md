# Wavetabler Fixes - Integration Guide

## Quick Start

I've created fixed versions of your most problematic files. Here's how to integrate them:

### Option 1: Replace Files Directly (Easiest)

1. **Backup your current files first!**
   ```bash
   cd /path/to/wavetabler
   mkdir backup
   cp *.py backup/
   ```

2. **Replace with fixed versions:**
   - `b_menu_fixed.py` → rename to `b_menu.py`
   - `cc_channels_fixed.py` → rename to `cc_channels.py`
   - `f_sort_fixed.py` → rename to `f_sort.py`
   - `h_interpolate_fixed.py` → rename to `h_interpolate.py`

3. **Add helper functions to aa_common.py:**
   - Open `aa_common_additions.py`
   - Copy all functions into your `aa_common.py` file
   - Replace `initialize_settings()` with `initialize_settings_enhanced()` and rename it back to `initialize_settings()`

### Option 2: Manual Integration (More Control)

If you want to understand and selectively apply fixes, follow the individual sections below.

---

## Fix #36: Filename Sanitization

### Problem
Trailing spaces in filenames cause `os.makedirs()` to fail.

### Files to Modify
- `b_menu.py`
- `aa_common.py`

### Steps

1. **Add to aa_common.py:**

```python
import re  # Add at top if not present

def sanitize_filename(filename):
    """Remove trailing spaces and invalid characters"""
    filename = filename.strip()
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    filename = filename.rstrip('.')
    return filename
```

2. **Modify b_menu.py, around line 104:**

```python
# Before
selected_files = [file_details[i - 1]['file_name'] for i in selected_indices]
aa_common._base = os.path.splitext(selected_files[0])[0]

# After
selected_files = [file_details[i - 1]['file_name'] for i in selected_indices]
selected_files = [aa_common.sanitize_filename(f) for f in selected_files]
base_name = os.path.splitext(selected_files[0])[0]
aa_common._base = aa_common.sanitize_filename(base_name)
```

---

## Fix #32: Stereo Channel Parsing

### Problem
Cannot handle mixed input like "1-2,4" (range + comma).

### File to Modify
- `cc_channels.py`

### Steps

1. **Add new parser function (around line 92, before `choose_channels`):**

```python
def parse_channel_selection(selections, total_channels):
    """Parse mixed range and comma input like '1-2,4'"""
    selected_indices = set()
    
    try:
        parts = selections.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_indices.update(range(start - 1, end))
            else:
                idx = int(part)
                selected_indices.add(idx - 1)
        
        selected_indices = {i for i in selected_indices if 0 <= i < total_channels}
        if not selected_indices:
            return None
        return sorted(list(selected_indices))
    except (ValueError, IndexError) as e:
        print(f"Error: {e}. Use format: 1 or 1,3 or 1-2 or 1-2,4")
        return None
```

2. **Replace the parsing logic in `choose_channels` (around line 130-136):**

```python
# Before
if '-' in selections:
    start, end = map(int, selections.split('-'))
    selected_channels = ordered_channels[start-1:end]
else:
    indices = map(int, selections.split(','))
    selected_channels = [ordered_channels[i-1] for i in indices]

# After
while True:
    selections = aa_common.input_with_defaults(...).strip()
    
    selected_indices = parse_channel_selection(selections, len(ordered_channels))
    if selected_indices is not None:
        selected_channels = [ordered_channels[i] for i in selected_indices]
        print(f"Selected channels: {', '.join(selected_channels)}")
        return {name: file_paths[name] for name in selected_channels}
    else:
        print("Invalid selection. Please try again.")
```

---

## Fix #22: Autocorrelation Tolerance Issues

### Problem
Autocorrelation mode yields zero normal segments due to tight tolerances.

### File to Modify
- `f_sort.py`
- `aa_common.py`

### Steps

1. **Add to aa_common.py:**

```python
def calculate_adaptive_tolerance_percentage(segment_lengths, base_tolerance=5.0):
    """Adaptive tolerance based on segment variance"""
    if isinstance(segment_lengths, dict):
        lengths = list(segment_lengths.values())
    else:
        lengths = list(segment_lengths)
    
    if not lengths:
        return base_tolerance
    
    std_dev = np.std(lengths)
    mean_length = np.mean(lengths)
    
    if mean_length == 0:
        return base_tolerance
    
    cv = std_dev / mean_length
    if cv > 0.05:
        adaptive_tolerance = base_tolerance * (1 + cv * 10)
    else:
        adaptive_tolerance = base_tolerance
    
    return min(adaptive_tolerance, 20.0)
```

2. **Modify `initialize_settings()` in aa_common.py:**

```python
global_settings = {
    # ... existing settings ...
    'percent_tolerance': 5,  # For zero-crossing
    'percent_tolerance_autocorr': 10,  # NEW: For autocorrelation
}
```

3. **Modify f_sort.py `run()` function (around line 192):**

```python
# After line: wavecycle_samples_target, wavecycle_samples_target_avg = autocorrelation_sort(wavecycle_samples)

# Add this:
if get_autocorrelation_flag():
    # Use wider tolerance for autocorrelation
    base_tolerance = 10
    settings['percent_tolerance'] = calculate_adaptive_tolerance_percentage(
        wavecycle_samples, base_tolerance
    )
    print(f"Using adaptive tolerance for autocorrelation: {settings['percent_tolerance']:.1f}%")
```

4. **Modify `autocorrelation_sort()` in f_sort.py (around line 13):**

```python
def autocorrelation_sort(wavecycle_samples):
    if get_autocorrelation_flag():
        # Use AVERAGE instead of MODE for autocorrelation
        wavecycle_samples_target_avg = calculate_average_wavecycle_length(wavecycle_samples)
        mode_interval = get_mode_interval()
        
        print(f"Autocorrelation -- using average: {wavecycle_samples_target_avg}")
        # Return average as primary target
        return wavecycle_samples_target_avg, wavecycle_samples_target_avg
    else:
        # Zero-crossing uses mode
        mode_interval = calculate_mode_wavecycle_length(wavecycle_samples)
        wavecycle_samples_target_avg = calculate_average_wavecycle_length(wavecycle_samples)
        return mode_interval, wavecycle_samples_target_avg
```

---

## Fix #28: Add Method Tags to Filenames

### Problem
Can't tell which method was used to create a wavetable.

### Files to Modify
- `aa_common.py`
- `j_wvtblr.py`

### Steps

1. **Add to aa_common.py:**

```python
def get_method_suffix():
    """Returns '_a' for autocorrelation or '_z' for zero-crossing"""
    global autocorrelation_flag
    return "_a" if autocorrelation_flag else "_z"

def get_timestamped_filename(base, suffix, extension=".wav"):
    """Generate filename with method tag and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = get_method_suffix()
    base = sanitize_filename(base)
    return f"{base}{method_tag}_{timestamp}_{suffix}{extension}"
```

2. **Modify j_wvtblr.py save functions:**

```python
# In save_pick(), save_chunk(), etc. - replace:
output_file = os.path.join(wavetables_folder, f"{base}_{timestamp}_{suffix}.wav")

# With:
output_file = os.path.join(wavetables_folder, 
    aa_common.get_timestamped_filename(base, suffix))
```

---

## Fix #30: _chunk_chunk Suffix Bug

### Problem
Suffix gets duplicated when file already contains it.

### File to Modify
- `aa_common.py`
- `j_wvtblr.py`

### Steps

1. **Add to aa_common.py:**

```python
def add_suffix_to_filename(filename, suffix):
    """Add suffix without duplicating"""
    base, ext = os.path.splitext(filename)
    
    if base.endswith(f"_{suffix}"):
        return filename  # Already has it
    
    return f"{base}_{suffix}{ext}"
```

2. **Use in j_wvtblr.py before saving:**

```python
# Before generating output filename
suffix = "chunk"
safe_base = aa_common.add_suffix_to_filename(base, suffix)
# Then use safe_base in filename generation
```

---

## Fix: Skip Unnecessary Interpolation

### Problem
Files already at 2048 samples get resampled anyway.

### File to Modify
- `h_interpolate.py`

### Steps

Replace the main loop in `run()`:

```python
for filename in sorted(os.listdir(seg_folder)):
    if not filename.endswith('.wav'):
        continue
    
    # ... discard logic ...
    
    input_path = os.path.join(seg_folder, filename)
    output_path = os.path.join(frames_folder, filename)
    
    data, sr = sf.read(input_path)
    current_length = len(data)
    
    # NEW: Check if already correct length
    if current_length == aa_common.wavecycle_size:
        print(f"✓ Copying {filename} (already 2048 samples)")
        shutil.copy2(input_path, output_path)
    else:
        print(f"↻ Interpolating {filename}: {current_length} → 2048")
        interpolated = aa_common.interpolate_seg(data, sr)
        sf.write(output_path, interpolated, sr, subtype='FLOAT')
```

---

## Fix: Zero-Crossing Alignment for Pick Mode

### Problem
Pick mode doesn't align to zero crossings, causing clicks.

### Files to Modify
- `aa_common.py`
- `j_wvtblr.py`

### Steps

1. **Add to aa_common.py:**

```python
def find_nearest_zero_crossing(data, sample_index, window=None, threshold=None):
    """Find nearest zero crossing to given index"""
    global zero_threshold, ZERO_WINDOW
    
    if window is None:
        window = ZERO_WINDOW
    if threshold is None:
        threshold = zero_threshold
    
    start = max(0, sample_index - window)
    end = min(len(data), sample_index + window)
    segment = data[start:end]
    
    crossings = []
    for i in range(1, len(segment)):
        if (segment[i-1] * segment[i]) <= 0:
            if abs(segment[i]) < threshold and abs(segment[i-1]) < threshold:
                crossings.append(start + i)
    
    if crossings:
        return min(crossings, key=lambda x: abs(x - sample_index))
    else:
        return sample_index
```

2. **Modify `pick_on_click()` in j_wvtblr.py (around line 38):**

```python
def pick_on_click(event, data, sr, fig, ax, total_samples, fixed_length, wavecycle_size):
    global selected_segment
    
    if event.inaxes != ax:
        return
    
    clicked_sample_index = int(event.xdata * sr)
    
    # NEW: Align to zero crossing
    aligned_index = aa_common.find_nearest_zero_crossing(data, clicked_sample_index)
    print(f"Clicked: {clicked_sample_index}, Aligned to zero-crossing: {aligned_index}")
    
    # Use aligned_index instead of clicked_sample_index from here on
    if aligned_index < wavecycle_size / 2:
        nearest_interval = 0
        # ... rest of logic ...
```

---

## Testing Your Fixes

### Test Case 1: Filename Sanitization
```python
# Create a test file with trailing space
echo "test" > "input/test_file .wav"
# Run wavetabler - should not crash on mkdir
```

### Test Case 2: Channel Selection
```bash
# When prompted for channel selection, try:
# "1-2,4" should select channels 1, 2, and 4
# "1,3" should select channels 1 and 3
# "2-4" should select channels 2, 3, and 4
```

### Test Case 3: Autocorrelation Tolerance
```bash
# Run with autocorrelation method
# Check output - should have some normal segments
# Should print adaptive tolerance percentage
```

### Test Case 4: Method Tags
```bash
# Run with zero-crossing
# Output should include "_z_" in filename
# Run with autocorrelation
# Output should include "_a_" in filename
```

### Test Case 5: Skip Interpolation
```bash
# Check console output during interpolation
# Should see "✓ Copying" for files already at 2048 samples
# Should see "↻ Interpolating" for files that need it
```

---

## Debugging Tips

### If you get "wavecycle_samples_target is 0":
- Check that `set_wavecycle_samples_target()` is called before `f_sort.run()`
- Verify autocorrelation flag is set correctly
- Print wavecycle_samples dictionary to see if segments were detected

### If no normal segments found:
- Check tolerance percentage (should be higher for autocorrelation)
- Print lower_bound and upper_bound values
- Print actual segment lengths to compare
- Try widening tolerance manually

### If segments have clicks:
- Verify zero-crossing alignment is enabled in pick mode
- Check ZERO_THRESHOLD_DB value (try -50 if -40 is too strict)
- Verify ZERO_WINDOW is large enough (try 64 instead of 32)

---

## Rollback Instructions

If something breaks:

```bash
# Restore from backup
cd /path/to/wavetabler
cp backup/*.py .
```

Or restore individual files:
```bash
cp backup/b_menu.py .
cp backup/cc_channels.py .
# etc.
```

---

## Support

If you encounter issues:

1. **Check the console output** - many fixes add informative print statements
2. **Verify global variables** - add print statements to check values
3. **Test one fix at a time** - easier to isolate problems
4. **Check file permissions** - ensure tmp folders can be created
5. **Verify dependencies** - scipy, numpy, soundfile versions

Good luck with your wavetable processing!
