# wvtbl — Wavetable Generator (C++)

Self-contained C++17 CLI tool that creates wavetables from any audio file for use in **Serum**, **Vital**, and other wavetable synthesizers.

No external dependencies — pure standard library.

## Build

```bash
cd cpp
cmake -S . -B build
cmake --build build
```

On Windows with MSVC:
```cmd
cd cpp
cmake -S . -B build
cmake --build build --config Release
```

The binary will be at `build/wvtbl` (or `build/Release/wvtbl.exe` on Windows).

## Usage

```
wvtbl <input.wav> [options]

Options:
  -n, --frames <N>     Frame count 1-4096       (default: 256)
  -m, --mode <mode>    slice | spectral         (default: slice)
  -o, --output <path>  Output WAV path
      --blend <N>      Loop crossfade samples   (default: 64)
      --no-serum       Omit Serum clm marker
  -h, --help           Show help
```

## Modes

| Mode       | Best for |
|------------|----------|
| `slice`    | Any audio — drums, texture, noise, speech. Evenly divides audio into N frames, resamples each to 2048 samples. |
| `spectral` | Melodic/tonal audio. Autocorrelation pitch detection extracts single-cycle waveforms per frame. |

## Examples

Basic (256-frame wavetable from a drum loop):
```bash
wvtbl drumloop.wav
```

Spectral mode from a synth pad:
```bash
wvtbl pad.wav -m spectral -n 128 -o pad_wt.wav
```

Minimal frames, no Serum marker:
```bash
wvtbl voice.wav -n 16 --no-serum
```

## Output

- **Format:** 32-bit float, mono WAV
- **Frame size:** 2048 samples per frame
- **Total samples:** N × 2048
- **Serum support:** Includes `clm` chunk marker by default for automatic recognition
- Drop the output `.wav` directly into Serum's or Vital's wavetable browser.

## Input

Accepts mono or stereo WAV files in 16-bit, 24-bit, or 32-bit PCM, and 32-bit IEEE float. Stereo files are mixed down to mono automatically.
