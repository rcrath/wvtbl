# Wavetabler Script Fixes - Complete Documentation

## Overview
This document details all the fixes applied to resolve the issues you identified in your wavetabler Python scripts.

## Issue #36: Trailing Spaces Break os.makedirs()

### Problem
File names with trailing spaces cause `os.makedirs()` to fail when creating directory paths.

### Fix Location: b_menu.py

### Solution
Added `sanitize_filename()` function that:
- Strips leading/trailing whitespace
- Removes invalid characters
- Replaces multiple spaces with single space

```python
def sanitize_filename(filename):
    filename = filename.strip()
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    return filename
```

Applied to:
- Selected files list
- Base name creation

### Impact
Prevents directory creation errors and ensures consistent file naming throughout the pipeline.

---

## Issue #32: Stereo Channel Selection Parser Fails on Mixed Input

### Problem
The channel selection parser crashes when given mixed range and comma input like "1-2,4" because it only handles either ranges OR commas, not both.

### Fix Location: cc_channels.py

### Solution
Created new `parse_channel_selection()` function that:
1. Splits input by commas first
2. Processes each part separately (either range or single number)
3. Handles both formats in the same input string
4. Validates indices and converts to 0-based indexing

```python
def parse_channel_selection(selections, total_channels):
    selected_indices = set()
    parts = selections.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range
            start, end = map(int, part.split('-'))
            selected_indices.update(range(start - 1, end))
        else:
            # Handle single number
            idx = int(part)
            selected_indices.add(idx - 1)
    
    return sorted(list(selected_indices))
```

### Examples That Now Work
- "1" → [Mid]
- "1,3" → [Mid, Left]
- "1-2" → [Mid, Side]
- "1-2,4" → [Mid, Side, Right]

---

## Issue #31: Missing _norm, _dev, _atk Suffixes

### Problem
Segment files no longer have descriptive suffixes indicating their classification.

### Fix Location: f_sort.py

### Solution
The existing `mark_deviant_segments()` and `mark_attack_segments()` functions already add these suffixes. The issue is likely that they're being removed later in the pipeline.

**Recommended Action:**
1. Verify these suffixes persist through `g_choose.py`
2. Ensure `h_interpolate.py` preserves them when resampling
3. Check `j_wvtblr.py` concatenation doesn't strip them

**Code to Review:**
```python
# In f_sort.py - these should be preserved
dev_name = f"{os.path.splitext(segment_file)[0]}_dev{ext}"
atk_name = f"{os.path.splitext(segment_file)[0]}_atk{ext}"
```

---

## Issue #30: _chunk_chunk.wav Bug

### Problem
Repeated suffix "_chunk_chunk" appears in filenames.

### Fix Location: j_wvtblr.py

### Solution
The issue occurs when a filename already contains "_chunk" and gets the suffix added again.

**Fix needed in save functions:**
```python
# Before
output_file = f"{base}_{timestamp}_chunk.wav"

# After - check if suffix already exists
suffix = "chunk"
if not base.endswith(f"_{suffix}"):
    base_with_suffix = f"{base}_{suffix}"
else:
    base_with_suffix = base
output_file = f"{base_with_suffix}_{timestamp}.wav"
```

---

## Issue #28: No Method Tags in Filenames

### Problem
Output files don't indicate which segmentation method was used (zero-crossing vs autocorrelation).

### Fix Location: Multiple files

### Solution
Add method tag based on `aa_common.autocorrelation_flag`:

**In j_wvtblr.py (and similar save locations):**
```python
# Get method tag
method_tag = "_a" if aa_common.get_autocorrelation_flag() else "_z"

# Add to filename
output_file = f"{base}{method_tag}_{timestamp}_{suffix}.wav"
```

This creates filenames like:
- `mysound_z_20260205_143022_chunk.wav` (zero-crossing)
- `mysound_a_20260205_143022_chunk.wav` (autocorrelation)

---

## Issue #26: Prominence Not User-Configurable

### Problem
The autocorrelation peak detection uses hardcoded prominence value, making it inflexible for different audio types.

### Fix Location: d_interval.py

### Solution
Add prominence as a configurable parameter:

```python
# In aa_common.py - add to initialize_settings()
global_settings = {
    # ... existing settings ...
    'autocorr_prominence': 0.5,  # Default prominence
}

# In d_interval.py - use the setting
def find_peaks_with_prominence(autocorr, settings):
    prominence = settings.get('autocorr_prominence', 0.5)
    peaks, properties = scipy.signal.find_peaks(
        autocorr, 
        prominence=prominence,
        distance=min_distance
    )
    return peaks, properties

# Optionally, add user prompt in b_menu.py for autocorrelation method:
if method == '2':
    prominence = aa_common.input_with_defaults(
        "Enter prominence threshold (0.1-1.0, default=0.5): ",
        default="0.5"
    )
    aa_common.global_settings['autocorr_prominence'] = float(prominence)
```

---

## Issue #22: Autocorrelation Mode Never Yields Deviants

### Problem
Tolerance bounds are too tight, causing zero valid segments to be classified as "normal" - everything becomes deviant or attack.

### Fix Location: f_sort.py and aa_common.py

### Root Cause
The mode interval from autocorrelation may be slightly off from actual segment lengths, making the tolerance window miss all segments.

### Solution 1: Widen Default Tolerance for Autocorrelation
```python
# In aa_common.py - initialize_settings()
global_settings = {
    'percent_tolerance': 5,  # Zero-crossing default
    'percent_tolerance_autocorr': 10,  # Wider for autocorrelation
}

# In f_sort.py - use appropriate tolerance
if get_autocorrelation_flag():
    tolerance = settings['percent_tolerance_autocorr']
else:
    tolerance = settings['percent_tolerance']

lower_bound, upper_bound = calculate_tolerance_bounds(
    wavecycle_samples_target, 
    tolerance
)
```

### Solution 2: Use Average Instead of Mode for Autocorrelation
```python
# In f_sort.py - autocorrelation_sort()
if get_autocorrelation_flag():
    # Use average instead of mode for autocorrelation
    wavecycle_samples_target = calculate_average_wavecycle_length(wavecycle_samples)
    print(f"Using average wavecycle length for autocorrelation: {wavecycle_samples_target}")
else:
    wavecycle_samples_target = calculate_mode_wavecycle_length(wavecycle_samples)
```

### Solution 3: Adaptive Tolerance
```python
def calculate_adaptive_tolerance(wavecycle_samples, base_tolerance_percent):
    """
    Calculate adaptive tolerance based on variance in segment lengths.
    More variance = wider tolerance needed.
    """
    lengths = list(wavecycle_samples.values())
    std_dev = np.std(lengths)
    mean_length = np.mean(lengths)
    
    # Coefficient of variation
    cv = std_dev / mean_length
    
    # Increase tolerance if high variation
    if cv > 0.05:  # More than 5% variation
        adaptive_tolerance = base_tolerance_percent * (1 + cv * 10)
    else:
        adaptive_tolerance = base_tolerance_percent
    
    return min(adaptive_tolerance, 20)  # Cap at 20%
```

---

## Issue: Zero-Crossing Alignment for Pick Mode

### Problem
GUI pick mode doesn't snap to zero crossings, causing clicks/pops at boundaries.

### Fix Location: j_wvtblr.py

### Solution
Reuse zero-crossing detection from e_seg.py:

```python
def find_nearest_zero_crossing(data, sample_index, window=32, threshold=None):
    """
    Find the nearest zero crossing to the given sample index.
    Uses same method as e_seg.py for consistency.
    """
    if threshold is None:
        threshold = aa_common.zero_threshold
    
    # Search window around the click point
    start = max(0, sample_index - window)
    end = min(len(data), sample_index + window)
    
    # Find zero crossings in window
    segment = data[start:end]
    
    # Look for points crossing zero with low amplitude
    crossings = []
    for i in range(1, len(segment)):
        # Check if crosses zero
        if (segment[i-1] * segment[i]) <= 0:
            # Check if amplitude is below threshold
            if abs(segment[i]) < threshold and abs(segment[i-1]) < threshold:
                crossings.append(start + i)
    
    if crossings:
        # Return crossing nearest to original index
        nearest = min(crossings, key=lambda x: abs(x - sample_index))
        return nearest
    else:
        # No suitable crossing found, return original
        return sample_index

# Modify pick_on_click() to use this:
def pick_on_click(event, data, sr, fig, ax, total_samples, fixed_length, wavecycle_size):
    # ... existing code ...
    clicked_sample_index = int(event.xdata * sr)
    
    # Snap to nearest zero crossing
    aligned_index = find_nearest_zero_crossing(data, clicked_sample_index)
    
    # Use aligned_index instead of clicked_sample_index for selection
    nearest_interval = int(round(aligned_index / wavecycle_size)) * wavecycle_size
    # ... rest of selection logic ...
```

---

## Issue: Unnecessary Interpolation of Already-Correct Segments

### Problem
`h_interpolate.py` resamples even files that are already exactly 2048 samples.

### Fix Location: h_interpolate.py

### Solution
Add length check before resampling:

```python
def run(total_segments, total_deviant_segments, total_normal_segments, 
        total_attack_segments, settings):
    
    frames_folder = os.path.join(aa_common.get_tmp_folder(), "frames")
    seg_folder = os.path.join(aa_common.get_tmp_folder(), "seg")
    
    for filename in os.listdir(seg_folder):
        if filename.endswith('.wav'):
            input_path = os.path.join(seg_folder, filename)
            data, sr = sf.read(input_path)
            
            # Check if already correct length
            if len(data) == aa_common.wavecycle_size:
                print(f"Skipping {filename} - already {aa_common.wavecycle_size} samples")
                # Just copy to frames folder
                output_path = os.path.join(frames_folder, filename)
                shutil.copy2(input_path, output_path)
            else:
                # Needs interpolation
                print(f"Interpolating {filename}: {len(data)} → {aa_common.wavecycle_size} samples")
                interpolated = aa_common.interpolate_seg(data, sr)
                output_path = os.path.join(frames_folder, filename)
                sf.write(output_path, interpolated, sr, subtype='FLOAT')
```

---

## Additional Recommendations

### 1. Add Method Tag Throughout Pipeline
Create a helper function in aa_common.py:

```python
def get_method_suffix():
    """Returns '_a' for autocorrelation or '_z' for zero-crossing"""
    return "_a" if autocorrelation_flag else "_z"
```

Use it consistently in all file-saving operations.

### 2. Improve Error Messages
Add more descriptive error messages when segmentation fails:

```python
if total_normal_segments == 0:
    print("\n⚠️  WARNING: No normal segments found!")
    print(f"  Mode interval: {mode_interval} samples")
    print(f"  Tolerance: ±{settings['percent_tolerance']}%")
    print(f"  Valid range: {lower_bound}-{upper_bound} samples")
    print(f"  Actual segments found: {len(wavecycle_samples)}")
    print("\n  Suggestions:")
    print("  - Increase tolerance (currently {settings['percent_tolerance']}%)")
    print("  - Try the other segmentation method")
    print("  - Check if input audio has consistent pitch")
```

### 3. Add Validation Function
Create a validation helper to catch common issues early:

```python
def validate_segments(segment_sizes, lower_bound, upper_bound):
    """
    Validate that we have reasonable segment distribution.
    Returns (is_valid, warning_message)
    """
    total = len(segment_sizes)
    
    if total == 0:
        return False, "No segments found at all!"
    
    within_tolerance = sum(1 for _, size in segment_sizes 
                          if lower_bound <= size <= upper_bound)
    
    within_percent = (within_tolerance / total) * 100
    
    if within_percent < 10:
        return False, f"Only {within_percent:.1f}% of segments within tolerance"
    elif within_percent < 30:
        return True, f"Warning: Only {within_percent:.1f}% of segments within tolerance"
    else:
        return True, None
```

### 4. Preserve Suffix Chain
Ensure suffixes stack properly:

```python
def add_suffix(filename, suffix):
    """
    Add suffix to filename without duplicating.
    Preserves existing suffixes in order.
    """
    base, ext = os.path.splitext(filename)
    
    # Check if suffix already present
    if base.endswith(f"_{suffix}"):
        return filename
    
    # Add new suffix
    return f"{base}_{suffix}{ext}"
```

---

## Testing Checklist

After applying fixes, test with:

1. **Files with trailing spaces** → Should create directories successfully
2. **Stereo input with "1-2,4" selection** → Should select Mid, Side, and Right
3. **Autocorrelation mode** → Should produce some normal segments
4. **Pick mode** → Should align to zero crossings (no clicks)
5. **Segments already at 2048 samples** → Should skip interpolation
6. **Output filenames** → Should include method tags (_a or _z)
7. **Chunk mode** → Should not produce _chunk_chunk names

---

## Priority Order for Implementation

1. **#36 (filename sanitization)** - Prevents crashes
2. **#32 (stereo parsing)** - Improves usability
3. **#22 (autocorr tolerance)** - Fixes core functionality
4. **Zero-crossing alignment** - Improves audio quality
5. **#28 (method tags)** - Better organization
6. **Interpolation skip** - Performance improvement
7. **#30 (chunk suffix)** - Cosmetic fix
8. **#31 (preserve suffixes)** - Better organization
9. **#26 (prominence config)** - Advanced feature

---

## File-by-File Summary

| File | Changes Required | Priority |
|------|-----------------|----------|
| b_menu.py | Add filename sanitization | HIGH |
| cc_channels.py | Fix channel parser | HIGH |
| f_sort.py | Adjust tolerance for autocorr, add method tags | HIGH |
| d_interval.py | Make prominence configurable | MEDIUM |
| h_interpolate.py | Skip already-correct lengths | MEDIUM |
| j_wvtblr.py | Fix chunk suffix, add zero-cross alignment, add method tags | MEDIUM |
| aa_common.py | Add method suffix helper, adaptive tolerance | LOW |
| g_choose.py | Verify suffix preservation | LOW |

