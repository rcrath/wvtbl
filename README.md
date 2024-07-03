# wvtbl, a wavetable processor

## Overview
This is a script to process a short mono sound file into a set of serum compatible wavetables, one group at the rate of 93.75 hz, which will create 256 frames of 2048 sample single cycle serum wavetables, and a second file that creates variant number of integer multiples or submultiples to give the same 256 frame 2048 sample at a pitch as close to the source file's pitch as possible.  These files are all date/time stamped so that they will always have unique filenames if you want to try different settings. The 93.75 Hz wavetables are named `(source filename)__94Hz__datetime.wav`in a folder of the same name as the source file. THe "closest pitch" files are named `(source filename)__closestpitch__datetime.wav`in the same place.

## Running wvtbl

### Install python 3.11, which includes pip. other versions will not work.  instructions vary by OS. Google.

### using your own input files
A few source files are included for testing.  For now, the files need to be mono I think (have not tried stereo. wavetables are mono).  To add your own input file, copy the file to the "source" folder where the script is. The script works most accurately with single note source files, but experimentation is the order of the day.  If you have a tuner of some sort, it helps to know the frequency of the note.  You can get this from Audacity: load your input file into audacity, select the file segment, got to `Effect`, `pitch and tempo`, `change pitch` and it will show you the estimated pitch on the left near the top. For a more precise reading, go to the `analysis` dropdown, `Paul Brossier, `Aubio Pitch detect:` that will give you a dense series of pitch estimates. Just guess which is the most common!

you can run on the defaults though without knowing the pitch and if the algorithm has a confidence level greater the 50%, it will process the pitch automatically. 

### set the environment

wvtbl is run from the command line on your operating system. `cmd` in windows, terminal in MacOS and Linux.  setup is simple, just copy and paste the code for the respective OS that follows.

To set up a Python virtual environment including the libraries from Block 1, follow these steps:

Create the virtual environment:

On Linux/macOS:

`python3 -m venv wvtbl_env`
On Windows:

`python -m venv wvtbl_win`

Activate the virtual environment:

On Linux/macOS:

`source wvtbl_env/bin/activate`

On Windows:

`wvtbl_win\Scripts\activate`

Run the following command to install the required libraries:

`pip install numpy scipy librosa soundfile aubio tensorflow crepe pydub`

Now your environment is set up with all the necessary libraries, and you should be able to run your script without module errors. Remember to activate the environment each time you open a new terminal. Make sure you are in the directory where the script is when you set the environment up.

To deactivate the virtual environment when you're done, simply run:

`deactivate`

This command will return you to your global Python environment.

### first run
from the prompt within the python environment you set up, run 

`python wvtbl.py`

and accept defaults.  The results will come back with some information and the output files in the folder named after your filename.

# without defaults:

answer `no` to the defaults question and you get a series of choices.

1. Freq estimate. Enter the note or frequency. enter to accept default of this and other individual settings. 

2. Deviation tolerance.  Higher percentages loosen up the criteria for marking "deviant" files. Deviant files are those which are of a length x% or more different from the average sample size we are aiming for.  Sometimes it helps to discard these, below, if you have a noisy signal and your results are not smooth. Remember this is a wavetable, not an audio file, so smooth is not necessarily better, just different.  

3. If the script labeled the unstable attack portion of the audio, you can discard this

4. You can discard deviant segments

5. Discard the good ones? Glitch out!

6. Cleanup: The default is to cleanup the files generated.  Skip it (answer "no") to see the single cycle and tmp files generated for debugging or curiosity.

### output

As mentioned, these go in the folder named after your input with the "94Hz" and "closest_pitch" infixes and unique timestamps.  In the "concat" folder, you will find the whole concatenated "94Hz" file and the whole "closest pitch" file, before dividing into wavetables.  This can be helpful in deciding if you want a smoother wave table by discarding the deviant cycles and to see how seriously it affects your source file when it is cut up and interpolating into equal length single cycles.

### use in serum and other wavetable synths

In Serum, click the edit wavetable pencil, and in the middle menu, choose import, audio (fixed frame).

For other wavetable synths, RTM for import.  You have a 2048samplex256frame wavetable.

### example output

Temporary location of [my generated wavetables](http://digix.manoa.hawaii.edu/rreplay/wavetables/).

Enjoy!
Rich Rath
Way Music / https://way.net/music plz give it a listen!
2024
