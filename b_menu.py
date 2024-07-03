
# b_menu.py

import os
import aa_common
from scipy.io import wavfile

def run():
    # Ensure start file and tmp folder are set up
    aa_common.decide_start_file()  # This should now work if decide_start_file is correctly defined in aa_common
    aa_common.ensure_tmp_folder()

    start_file = aa_common.get_start_file()
    sample_rate, start_file_data = wavfile.read(start_file)
    base = aa_common.get_base()
    os.makedirs(base, exist_ok=True)

if __name__ == "__main__":
    run()
