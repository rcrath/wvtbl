import os
import shutil
from aa_common import get_tmp_folder

def perform_cleanup():
    """
    Performs cleanup by deleting the tmp folder.
    """
    tmp_folder = get_tmp_folder()

    try:
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
            print(f"Deleted folder: {tmp_folder}")

        print("Cleanup completed.")
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

def run():
    perform_cleanup()

if __name__ == "__main__":
    run()
