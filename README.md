# wvtbl, a wavetable processor

## Overview
This is a script to process short mono sound file into a set of serum compatible wavetables, one at the rate of 93.75 hz, which will create 256 frames of 2048 sample single cycle serum wavetables, and a secod file that creates variant number of integer multiples or submultiples to give the same 256 frame 2048 sample at a pitch as close to the input pitchj as possible.  These files are all date/time stamped so that they will always have unique filenamnes if you want to try different settings.the 93.75 HZ wavetables are named with the input file name + "_94Hz_ + datetime.wav in a folder of the same name as the source file."  

## Running wvtble

### Install python 3.11, which includes pip. other versions will not work.  instructions vary by OS

### using your own input files
A few source files are included for testing.  To add your own input file, copy the file to the "source" folder where the script is. the script works most accurately with single not inout files, but experimentation is the order of the day.  If you have a tuner of some sort, it helps to know the frequncy of the note.  You can get this from Audacity: load your input file into audacity, select the file segment, got to `Effect', `pitch and tempo`, change pitch and it will show you the estimated pitch on the left near the top. For a more precise reading, go to the `analysis` dropdown, `Paul Brossier`, `Aubio Pitch detect:` that will give you a dense series of pitch estimates. guess which is the most common!

you can run on the defaults though without knowing the pitch and if the algorithm has a confidence level greater the 50% sure, it will process the pitch automatically. 
### set the environment

wvtble is run from the command line on your operating system. `cmd` in windows, terminal in MacOS and Linux.  setup is simple, just copy and paste the code for the respective OS that follows.

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

Now your environment is set up with all the necessary libraries, and you should be able to run your script without module errors. Remember to activate the environment each time before you work.

To deactivate the virtual environment when you're done, simply run:

`deactivate`

This command will return you to your global Python environment.

### first run
from the prompt within the python environment you set up, run 

`python wvtbl.py`

and accept defaults.  The results will come back with some information and the output files in the folder named after you filename.

# without defaults:

answer `no` to the defaults question and you get a series of choices.

1. freq estimate. enter the note or frequency. enter to accept default o this and other settings. 

2. deviation tolerance.  higher percentages loosen up the criteria for marking "deviant" files. Deviant files are those which are of a lenght x% or more different from the average sample size we are aiming for.  Sometimes it helps to discard these, below, if you have a noisy signal and your results are not smooth. remember this is a wavetable, not an audio file, so smooth is not necessarily better, just different.  

3. if the script labeled the unstable attack portion of the audio, you can discard this

4. you can discard deviant segments

5. discard the good ones!?

6. cleanup: default is to cleanup the files generated.  skip it (answer "no") to see the single cycle and tmp files generated for debugging or curiosity.

### output

As mentioned, these go in the folder named after your input with the "94Hz" and "closest pitch" infixes and unique timestamps.  in the "concat" folder, you will find the whole concatenated pitch shifted and interpolated file.  this is helpful in deciding if you want a smoother wave table by discarding the deviant cycles.

Enjoy!
Rich Rath
Way Music / https://way.net/music plz give it a listen!
2024

### use in serum and other wavetable synths

In Serum, click the edit wavetable pencil, and in the middle menu, choose import, audio (fixed frame).

For other wavetable synths, RTM for import.  You have a 2048samplex256frame wavetable.
