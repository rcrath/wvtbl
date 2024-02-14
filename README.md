# wvtbl, a wavetable processor

## Overview
This is a script to process a short mono sound file into a set of serum compatible wavetables, one group at the rate of 93.75 hz, which will create 256 frames of 2048 sample single cycle serum wavetables, and a second file that creates variant number of integer multiples or submultiples to give the same 256 frame 2048 sample at a pitch as close to the source file's pitch as possible.  These files are all date/time stamped so that they will always have unique filenames if you want to try different settings. The 93.75 Hz wavetables are named `(source filename)__94Hz__datetime.wav`in a folder of the same name as the source file. THe "closest pitch" files are named `(source filename)__closestpitch__datetime.wav`in the same place.

## Running wvtbl

### Install python 3.11, which includes pip. other versions will not work.  instructions vary by OS. Google.

Python 3.11 download link, for convenience:

`https://www.python.org/downloads/release/python-3118/`

Here's the specific recommended Windows installer link:

`https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe`

...And the same for MacOS 10.9 and later:

`https://www.python.org/ftp/python/3.11.8/python-3.11.8-macos11.pkg`

### using your own input files

A few source files are included for testing.  For now, the files need to be mono I think (have not tried stereo. wavetables are mono).  To add your own input file, copy the file to the "source" folder where the script is. The script works most accurately with single note source files, but experimentation is the order of the day.  If you have a tuner of some sort, it helps to know the frequency of the note.  You can get this from Audacity: load your input file into audacity, select the file segment, got to `Effect`, `pitch and tempo`, `change pitch` and it will show you the estimated pitch on the left near the top. For a more precise reading, go to the `analysis` dropdown, `Paul Brossier, `Aubio Pitch detect:` that will give you a dense series of pitch estimates. Just guess which is the most common!

you can run on the defaults though without knowing the pitch and if the algorithm has a confidence level greater the 50%, it will process the pitch automatically. 

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

Now your environment is set up with all the necessary libraries, and you should be able to run your script without module errors. Remember to activate the environment each time you open a new terminal. Make sure you are in the directory where the script is when you set the environment up.

To deactivate the virtual environment when you're done, simply run:

`deactivate`

This command will return you to your global Python environment.

### first run
from the prompt within the python environment you set up, put the file you want to work on in the source folder and run 

`python wvtbl.py`

choose a file from the list and accept defaults.  The results will come back with some information and the output files in the folder named after your filename.

### without defaults:

answer `no` to the defaults question and you get a series of choices.

1. Freq estimate. Enter the note or frequency. enter to accept default of this and other individual settings. 

2. Deviation tolerance.  Higher percentages loosen up the criteria for marking "deviant" files. Deviant files are those which are of a length x% or more different from the average sample size we are aiming for.  Sometimes it helps to discard these, below, if you have a noisy signal and your results are not smooth. Remember this is a wavetable, not an audio file, so smooth is not necessarily better, just different.  __2DO__? there is output saying how many deviant samples.  I will improve this to show haw many good segments, attack segments, and total segments too.  You generally need 256 segments for the script to work though __2DO__ I want to provide an option for back filling for low numbers of segments or inserting silence to process percussive sounds. 

3. If the script labeled the unstable attack portion of the audio, you can discard this if you like. this will remove the attack phase from the first 256 frame wavetables. 

4. You can discard deviant segments. if there are enough good segments this can really smooth out your output. if not, raise the tolerance and try again. 

5. Discard the good ones? Glitch out!

6. Cleanup: The default is to cleanup the files generated.  Skip it (answer "no") to see the single cycle and tmp files generated for debugging or curiosity. 

### output

As mentioned, these go in the folder named after your input with the "94Hz" and "closest_pitch" infixes and unique timestamps.  In the "concat" folder, you will find the whole concatenated "94Hz" file and the whole "closest pitch" file, before dividing into wavetables.  This can be helpful in deciding if you want a smoother wave table by discarding the deviant cycles and to see how seriously it affects your source file when it is cut up and interpolating into equal length single cycles.

### use in serum and other wavetable synths

In Serum, click the edit wavetable pencil, and in the middle menu, choose import, audio (fixed frame) and enter 2048 for the frame length. 

For other wavetable synths, RTM for import.  You have a 2048 sample * 256 frame wavetable to work with.

### example output

Temporary location of [my generated wavetables](http://digix.manoa.hawaii.edu/rreplay/wavetables/).

# how it works. 

Anyone interested, here is the breakdown of what it does:

* upsample source file to 192k and put 2048 sample fade on start and end of upsampled file.
* Do pitch recognition using neural net or input your estimate, and calculate the integer number of samples nearest that pitch. @Ruoho Ruotsi has another approach I want to try based on correlating peaks instead of finding zero crossings. I have a few other ideas for this part as well.

* From there, work with integer samples per cycle, which saves trouble with aliasing instead of working with float of pitch since our goal is a wavecycle of n samples that is closest to the pitch estimate.

* define a tolerance +/-% and find all the rising zero crossings, save as single wavecycle segments and interpolate the the segment size of all the segments that are within the tolerance range to match the number of samples in your pitch estimate. You can do all segments or just the ones within the tolerance.

* These then get interpolated to 2048 samples long at 192k, which is where the 93.75 Hz comes from.

* the initial same size segments are also interpolated to the nearest power of 2 samples long segments.

** If the pitch result is higher, multiple wave cycles can fit in one 2048 sample evenly.
** If the pitch is lower, one wavecycle will not fit so you have to downsample to 96k if the nearest 192h power of 2 is 4096 samples of long, which makes a single 46.875 Hz segment fit into 1 2048 sample wavecycle (96000/2048= 46.875). Then to 48k if your nearest wavecycle size is 8192 samples long (48000/2048=23.438Hz) for if you put in a sub bass source file in the 20s.
* Then you take each batch of segments and combine them back into two reconstructions of the input file, one at 93.75Hz and the other at the nearest power of 2 to the original pitch.
* Then those get chopped into 256 frames of 2048 sample/frame serum wavetables sized files, one at 93.75 Hz and the other closer to the original pitch of the source file.
* Bonus: the two full files are saved in concat folder to help you refine the non-default settings.

Enjoy!
Rich Rath
Way Music / https://way.net/music plz give it a listen!
2024
