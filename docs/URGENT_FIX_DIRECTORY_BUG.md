# URGENT FIX: Directory Creation Bug

## The Problem

Your error:
```
FileNotFoundError: [Errno 2] No such file or directory: 'tmp\\cpy'
```

This happens because `b_menu.py` deletes the tmp folder but doesn't recreate the `cpy` subfolder before trying to copy files to it.

## The Quick Fix

In your `b_menu.py`, find this section (around lines 144-160):

```python
# Check if tmp folder exists before proceeding
tmp_folder = aa_common.get_tmp_folder()
if os.path.exists(tmp_folder):
    cleanup_choice = aa_common.input_with_defaults("Tmp folder exists, remove? (Y or ENTER / n to quit): ").strip().lower() or 'y'

    if cleanup_choice == 'y':
        aa_common.perform_cleanup()  # Call cleanup function from aa_common
    else:
        print("Quitting script.")
        exit()  # Quit the script if the user chooses 'n'
else:
    # If the tmp folder does not exist, create it
    aa_common.ensure_tmp_folder()

for file_name in selected_files:
    source_file_path = os.path.join(aa_common.source_folder, file_name)
    shutil.copy2(source_file_path, cpy_folder)
```

**Replace it with this:**

```python
# Check if tmp folder exists before proceeding
tmp_folder = aa_common.get_tmp_folder()
if os.path.exists(tmp_folder):
    cleanup_choice = aa_common.input_with_defaults("Tmp folder exists, remove? (Y or ENTER / n to quit): ").strip().lower() or 'y'

    if cleanup_choice == 'y':
        aa_common.perform_cleanup()  # Call cleanup function from aa_common
    else:
        print("Quitting script.")
        exit()

# FIX: Always ensure tmp folder and subdirectories exist after cleanup
aa_common.ensure_tmp_folder()

# FIX: Also ensure the cpy subfolder exists
actual_cpy_folder = os.path.join(tmp_folder, "cpy")
if not os.path.exists(actual_cpy_folder):
    os.makedirs(actual_cpy_folder, exist_ok=True)
    print(f"Created cpy folder: {actual_cpy_folder}")

# Now copy files
for file_name in selected_files:
    source_file_path = os.path.join(aa_common.source_folder, file_name)
    dest_file_path = os.path.join(actual_cpy_folder, file_name)
    print(f"Copying {source_file_path} to {dest_file_path}")
    shutil.copy2(source_file_path, dest_file_path)
```

## What Changed

1. **Removed the `else` clause** - We now ALWAYS call `ensure_tmp_folder()` after cleanup
2. **Added explicit cpy folder creation** - Creates `tmp/cpy` directory before copying
3. **Fixed the copy destination** - Uses `dest_file_path` instead of just `cpy_folder`

## Why This Happened

The original logic was:
- IF tmp exists → delete it → don't recreate subdirectories
- ELSE tmp doesn't exist → create it

But after deletion, the subdirectories (like `cpy`) weren't being recreated, causing the copy to fail.

## Test It

After making this change, run your script again. It should now:
1. Delete the old tmp folder (if it exists)
2. Recreate the tmp folder
3. Create the cpy subfolder
4. Successfully copy files to tmp/cpy

You should see output like:
```
Deleted folder: flange220\tmp
Created cpy folder: flange220\tmp\cpy
Copying E:\...\input\your_file.wav to flange220\tmp\cpy\your_file.wav
```
