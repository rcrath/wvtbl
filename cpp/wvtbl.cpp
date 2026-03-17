// wvtbl.cpp — standalone wavetable generator for Serum / Vital / etc.
// C++17, no external dependencies. Reads any mono/stereo WAV (16/24/32-bit or float),
// outputs a 32-bit float mono WAV suitable for drag-and-drop into wavetable synths.
//
// Two extraction modes:
//   slice    — evenly divides audio into N frames, resamples each to 2048 samples
//   spectral — autocorrelation pitch detection, extracts single-cycle waveforms
//
// Build:  cmake -S . -B build && cmake --build build
// Usage:  wvtbl <input.wav> [options]

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// WAV structures
// ============================================================================

#pragma pack(push, 1)
struct RiffHeader {
    char     riff[4];       // "RIFF"
    uint32_t fileSize;
    char     wave[4];       // "WAVE"
};

struct ChunkHeader {
    char     id[4];
    uint32_t size;
};

struct FmtChunk {
    uint16_t audioFormat;   // 1 = PCM, 3 = IEEE float
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};
#pragma pack(pop)

// ============================================================================
// WAV reader — supports 16/24/32-bit PCM and 32-bit float, mono or stereo
// ============================================================================

struct WavData {
    std::vector<float> samples;   // mono, normalized to [-1, 1]
    uint32_t sampleRate = 0;
};

static bool readWav(const std::string& path, WavData& wav) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "error: cannot open " << path << "\n";
        return false;
    }

    RiffHeader rh{};
    if (fread(&rh, sizeof(rh), 1, f) != 1 ||
        memcmp(rh.riff, "RIFF", 4) != 0 ||
        memcmp(rh.wave, "WAVE", 4) != 0) {
        std::cerr << "error: not a valid WAV file\n";
        fclose(f);
        return false;
    }

    FmtChunk fmt{};
    bool foundFmt = false, foundData = false;
    std::vector<uint8_t> rawData;

    while (!feof(f)) {
        ChunkHeader ch{};
        if (fread(&ch, sizeof(ch), 1, f) != 1) break;

        if (memcmp(ch.id, "fmt ", 4) == 0) {
            size_t toRead = std::min<size_t>(ch.size, sizeof(fmt));
            if (fread(&fmt, toRead, 1, f) != 1) break;
            if (ch.size > toRead) fseek(f, (long)(ch.size - toRead), SEEK_CUR);
            foundFmt = true;
        } else if (memcmp(ch.id, "data", 4) == 0) {
            rawData.resize(ch.size);
            if (fread(rawData.data(), 1, ch.size, f) != ch.size) {
                rawData.resize(0);
            }
            foundData = true;
        } else {
            // skip unknown chunks
            fseek(f, (long)ch.size, SEEK_CUR);
        }

        // chunks are word-aligned
        if (ch.size & 1) fseek(f, 1, SEEK_CUR);
    }
    fclose(f);

    if (!foundFmt || !foundData || rawData.empty()) {
        std::cerr << "error: incomplete WAV file\n";
        return false;
    }
    if (fmt.audioFormat != 1 && fmt.audioFormat != 3) {
        std::cerr << "error: unsupported WAV format (need PCM or IEEE float)\n";
        return false;
    }

    wav.sampleRate = fmt.sampleRate;
    size_t bytesPerSample = fmt.bitsPerSample / 8;
    size_t frameBytes = bytesPerSample * fmt.numChannels;
    size_t numFrames = rawData.size() / frameBytes;

    wav.samples.resize(numFrames);
    const uint8_t* ptr = rawData.data();

    for (size_t i = 0; i < numFrames; i++) {
        double acc = 0.0;
        for (uint16_t ch = 0; ch < fmt.numChannels; ch++) {
            double val = 0.0;
            if (fmt.audioFormat == 3 && fmt.bitsPerSample == 32) {
                float fv;
                memcpy(&fv, ptr, 4);
                val = fv;
            } else if (fmt.bitsPerSample == 16) {
                int16_t sv;
                memcpy(&sv, ptr, 2);
                val = sv / 32768.0;
            } else if (fmt.bitsPerSample == 24) {
                int32_t sv = (int32_t)ptr[0] | ((int32_t)ptr[1] << 8) | ((int32_t)ptr[2] << 16);
                if (sv & 0x800000) sv |= ~0xFFFFFF;  // sign extend
                val = sv / 8388608.0;
            } else if (fmt.bitsPerSample == 32 && fmt.audioFormat == 1) {
                int32_t sv;
                memcpy(&sv, ptr, 4);
                val = sv / 2147483648.0;
            }
            ptr += bytesPerSample;
            acc += val;
        }
        wav.samples[i] = (float)(acc / fmt.numChannels);  // mixdown to mono
    }

    return true;
}

// ============================================================================
// WAV writer — 32-bit float mono with optional Serum clm marker
// ============================================================================

static bool writeWav(const std::string& path, const std::vector<float>& samples,
                     uint32_t sampleRate, int numFrames, bool writeClm) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "error: cannot create " << path << "\n";
        return false;
    }

    // clm chunk: Serum wavetable marker
    // format: "clm " chunk with text "<!>2048 0 0 0\0" (frame size, flags)
    std::string clmText;
    if (writeClm) {
        clmText = "<!>2048 0 0 " + std::to_string(numFrames);
        clmText.push_back('\0');
    }

    uint32_t dataSize = (uint32_t)(samples.size() * sizeof(float));
    uint32_t clmChunkSize = writeClm ? (uint32_t)(8 + clmText.size()) : 0;

    FmtChunk fmt{};
    fmt.audioFormat   = 3;  // IEEE float
    fmt.numChannels   = 1;
    fmt.sampleRate    = sampleRate;
    fmt.bitsPerSample = 32;
    fmt.blockAlign    = 4;
    fmt.byteRate      = sampleRate * 4;

    uint32_t fmtChunkSize = sizeof(FmtChunk);
    uint32_t totalSize = 4 + (8 + fmtChunkSize) + (8 + dataSize) + clmChunkSize;

    // RIFF header
    RiffHeader rh{};
    memcpy(rh.riff, "RIFF", 4);
    rh.fileSize = totalSize;
    memcpy(rh.wave, "WAVE", 4);
    fwrite(&rh, sizeof(rh), 1, f);

    // fmt chunk
    ChunkHeader ch{};
    memcpy(ch.id, "fmt ", 4);
    ch.size = fmtChunkSize;
    fwrite(&ch, sizeof(ch), 1, f);
    fwrite(&fmt, sizeof(fmt), 1, f);

    // data chunk
    memcpy(ch.id, "data", 4);
    ch.size = dataSize;
    fwrite(&ch, sizeof(ch), 1, f);
    fwrite(samples.data(), sizeof(float), samples.size(), f);

    // clm chunk (Serum wavetable marker)
    if (writeClm) {
        memcpy(ch.id, "clm ", 4);
        ch.size = (uint32_t)clmText.size();
        fwrite(&ch, sizeof(ch), 1, f);
        fwrite(clmText.data(), 1, clmText.size(), f);
        if (clmText.size() & 1) {
            uint8_t pad = 0;
            fwrite(&pad, 1, 1, f);
        }
    }

    fclose(f);
    return true;
}

// ============================================================================
// DSP helpers
// ============================================================================

static constexpr int FRAME_SIZE = 2048;

// Linear interpolation resample from srcLen samples to dstLen samples
static std::vector<float> resample(const float* src, size_t srcLen, size_t dstLen) {
    std::vector<float> dst(dstLen);
    if (srcLen == 0 || dstLen == 0) return dst;
    if (srcLen == 1) {
        std::fill(dst.begin(), dst.end(), src[0]);
        return dst;
    }
    double ratio = (double)(srcLen - 1) / (double)(dstLen - 1);
    for (size_t i = 0; i < dstLen; i++) {
        double pos = i * ratio;
        size_t idx = (size_t)pos;
        double frac = pos - idx;
        if (idx + 1 < srcLen)
            dst[i] = (float)(src[idx] * (1.0 - frac) + src[idx + 1] * frac);
        else
            dst[i] = src[srcLen - 1];
    }
    return dst;
}

// Normalize a frame to peak amplitude [-1, 1]
static void normalize(std::vector<float>& buf) {
    float peak = 0.0f;
    for (auto s : buf) peak = std::max(peak, std::fabs(s));
    if (peak > 1e-8f) {
        float gain = 1.0f / peak;
        for (auto& s : buf) s *= gain;
    }
}

// Apply crossfade blend at loop boundaries for click-free looping
static void applyBlend(std::vector<float>& frame, int blendSamples) {
    int n = (int)frame.size();
    if (blendSamples <= 0 || blendSamples > n / 2) return;
    for (int i = 0; i < blendSamples; i++) {
        float t = (float)i / (float)blendSamples;
        // Blend end into start and start into end using equal-power crossfade
        float fadeIn  = std::sqrt(t);
        float fadeOut = std::sqrt(1.0f - t);
        float startVal = frame[i];
        float endVal   = frame[n - blendSamples + i];
        frame[i]                    = startVal * fadeIn + endVal * fadeOut;
        frame[n - blendSamples + i] = endVal * fadeIn + startVal * fadeOut;
    }
}

// ============================================================================
// Autocorrelation pitch detection
// ============================================================================

static double detectPitch(const float* buf, size_t len, uint32_t sampleRate) {
    // Autocorrelation method
    // Search for fundamental between ~30 Hz and ~4000 Hz
    int minLag = std::max(2, (int)(sampleRate / 4000.0));
    int maxLag = std::min((int)(len / 2), (int)(sampleRate / 30.0));

    if (maxLag <= minLag) return 0.0;

    // Compute normalized autocorrelation
    double bestCorr = -1.0;
    int bestLag = minLag;

    // Energy of the window
    double energy0 = 0.0;
    for (int i = 0; i < maxLag; i++) {
        energy0 += (double)buf[i] * buf[i];
    }

    for (int lag = minLag; lag <= maxLag; lag++) {
        double corr = 0.0;
        double energy1 = 0.0;
        for (int i = 0; i < maxLag; i++) {
            corr += (double)buf[i] * buf[i + lag];
            energy1 += (double)buf[i + lag] * buf[i + lag];
        }
        double denom = std::sqrt(energy0 * energy1);
        if (denom > 1e-12) {
            double norm = corr / denom;
            if (norm > bestCorr) {
                bestCorr = norm;
                bestLag = lag;
            }
        }
    }

    if (bestCorr < 0.3) return 0.0;  // no clear pitch found

    // Parabolic interpolation for sub-sample accuracy
    if (bestLag > minLag && bestLag < maxLag) {
        auto corrAt = [&](int lag) -> double {
            double c = 0.0, e = 0.0;
            for (int i = 0; i < maxLag; i++) {
                c += (double)buf[i] * buf[i + lag];
                e += (double)buf[i + lag] * buf[i + lag];
            }
            double d = std::sqrt(energy0 * e);
            return d > 1e-12 ? c / d : 0.0;
        };
        double y0 = corrAt(bestLag - 1);
        double y1 = bestCorr;
        double y2 = corrAt(bestLag + 1);
        double shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2);
        if (std::isfinite(shift) && std::fabs(shift) < 1.0) {
            return (double)sampleRate / (bestLag + shift);
        }
    }

    return (double)sampleRate / bestLag;
}

// ============================================================================
// Slice mode — evenly divide audio into N frames
// ============================================================================

static std::vector<float> extractSlice(const WavData& wav, int numFrames, int blendSamples) {
    std::vector<float> output;
    output.reserve(numFrames * FRAME_SIZE);

    size_t totalSamples = wav.samples.size();
    double sliceLen = (double)totalSamples / numFrames;

    for (int i = 0; i < numFrames; i++) {
        size_t start = (size_t)(i * sliceLen);
        size_t end   = (size_t)((i + 1) * sliceLen);
        if (end > totalSamples) end = totalSamples;
        size_t len = end - start;
        if (len == 0) {
            // pad with silence
            for (int j = 0; j < FRAME_SIZE; j++) output.push_back(0.0f);
            continue;
        }

        auto frame = resample(&wav.samples[start], len, FRAME_SIZE);
        normalize(frame);
        applyBlend(frame, blendSamples);
        output.insert(output.end(), frame.begin(), frame.end());
    }

    return output;
}

// ============================================================================
// Spectral mode — extract single-cycle waveforms using pitch detection
// ============================================================================

static std::vector<float> extractSpectral(const WavData& wav, int numFrames, int blendSamples) {
    std::vector<float> output;
    output.reserve(numFrames * FRAME_SIZE);

    size_t totalSamples = wav.samples.size();

    // Analyze pitch across the file in windows
    // We'll take numFrames evenly-spaced positions and extract one cycle at each
    double step = (double)totalSamples / (numFrames + 1);
    size_t analysisWindow = std::min<size_t>(totalSamples, (size_t)(wav.sampleRate / 15));  // enough for ~15 Hz

    int successCount = 0;
    int failCount = 0;

    for (int i = 0; i < numFrames; i++) {
        size_t center = (size_t)((i + 1) * step);

        // Build analysis window around center
        size_t winStart = 0;
        if (center > analysisWindow / 2) winStart = center - analysisWindow / 2;
        size_t winEnd = winStart + analysisWindow;
        if (winEnd > totalSamples) {
            winEnd = totalSamples;
            if (winEnd > analysisWindow) winStart = winEnd - analysisWindow;
            else winStart = 0;
        }

        size_t winLen = winEnd - winStart;
        double pitch = detectPitch(&wav.samples[winStart], winLen, wav.sampleRate);

        std::vector<float> frame;
        if (pitch > 20.0) {
            // Extract one cycle at the detected pitch
            double cycleSamples = wav.sampleRate / pitch;
            size_t cycleLen = (size_t)(cycleSamples + 0.5);

            // Find a good extraction point near center (zero crossing)
            size_t extractStart = center;
            if (extractStart + cycleLen > totalSamples) {
                if (totalSamples > cycleLen)
                    extractStart = totalSamples - cycleLen;
                else
                    extractStart = 0;
            }

            // Search for zero crossing near extractStart
            size_t searchRange = std::min<size_t>(cycleLen / 2, 200);
            size_t bestZC = extractStart;
            float bestDist = 1e9f;
            for (size_t j = 0; j < searchRange && extractStart + j + cycleLen <= totalSamples; j++) {
                size_t pos = extractStart + j;
                float val = std::fabs(wav.samples[pos]);
                // Prefer positive-going zero crossings
                bool posGoing = (pos + 1 < totalSamples) && wav.samples[pos] <= 0.0f && wav.samples[pos + 1] > 0.0f;
                float dist = posGoing ? val * 0.1f : val;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestZC = pos;
                }
            }
            extractStart = bestZC;

            size_t extractLen = std::min<size_t>(cycleLen, totalSamples - extractStart);
            frame = resample(&wav.samples[extractStart], extractLen, FRAME_SIZE);
            successCount++;
        } else {
            // Pitch detection failed — fall back to slice at this position
            size_t fallbackLen = std::min<size_t>((size_t)(wav.sampleRate / 100), totalSamples);
            size_t fStart = center;
            if (fStart + fallbackLen > totalSamples) {
                fStart = (totalSamples > fallbackLen) ? totalSamples - fallbackLen : 0;
            }
            size_t fLen = std::min<size_t>(fallbackLen, totalSamples - fStart);
            frame = resample(&wav.samples[fStart], fLen, FRAME_SIZE);
            failCount++;
        }

        normalize(frame);
        applyBlend(frame, blendSamples);
        output.insert(output.end(), frame.begin(), frame.end());
    }

    if (failCount > 0) {
        std::cerr << "note: pitch detection succeeded for " << successCount
                  << "/" << numFrames << " frames (fallback used for "
                  << failCount << ")\n";
    }

    return output;
}

// ============================================================================
// CLI
// ============================================================================

static void printUsage(const char* prog) {
    std::cout << "wvtbl — wavetable generator for Serum / Vital / etc.\n\n"
              << "Usage: " << prog << " <input.wav> [options]\n\n"
              << "Options:\n"
              << "  -n, --frames <N>     Frame count 1-4096       (default: 256)\n"
              << "  -m, --mode <mode>    slice | spectral         (default: slice)\n"
              << "  -o, --output <path>  Output WAV path\n"
              << "      --blend <N>      Loop crossfade samples   (default: 64)\n"
              << "      --no-serum       Omit Serum clm marker\n"
              << "  -h, --help           Show this help\n\n"
              << "Modes:\n"
              << "  slice     Evenly divides audio into N frames. Works with any audio.\n"
              << "  spectral  Pitch detection extracts single-cycle waveforms. Best for\n"
              << "            tonal/melodic audio.\n\n"
              << "Output: 32-bit float mono WAV, N x 2048 samples.\n"
              << "        Drop directly into Serum or Vital's wavetable browser.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath;
    std::string outputPath;
    std::string mode = "slice";
    int numFrames = 256;
    int blendSamples = 64;
    bool writeClm = true;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-n" || arg == "--frames") {
            if (++i >= argc) { std::cerr << "error: --frames needs a value\n"; return 1; }
            numFrames = std::atoi(argv[i]);
        } else if (arg == "-m" || arg == "--mode") {
            if (++i >= argc) { std::cerr << "error: --mode needs a value\n"; return 1; }
            mode = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { std::cerr << "error: --output needs a value\n"; return 1; }
            outputPath = argv[i];
        } else if (arg == "--blend") {
            if (++i >= argc) { std::cerr << "error: --blend needs a value\n"; return 1; }
            blendSamples = std::atoi(argv[i]);
        } else if (arg == "--no-serum") {
            writeClm = false;
        } else if (arg[0] == '-') {
            std::cerr << "error: unknown option " << arg << "\n";
            return 1;
        } else {
            inputPath = arg;
        }
    }

    if (inputPath.empty()) {
        std::cerr << "error: no input file specified\n";
        return 1;
    }

    // Validate
    if (numFrames < 1 || numFrames > 4096) {
        std::cerr << "error: frame count must be 1-4096\n";
        return 1;
    }
    if (mode != "slice" && mode != "spectral") {
        std::cerr << "error: mode must be 'slice' or 'spectral'\n";
        return 1;
    }
    if (blendSamples < 0 || blendSamples > FRAME_SIZE / 2) {
        std::cerr << "error: blend must be 0-" << FRAME_SIZE / 2 << "\n";
        return 1;
    }

    // Default output path
    if (outputPath.empty()) {
        size_t dot = inputPath.rfind('.');
        std::string base = (dot != std::string::npos) ? inputPath.substr(0, dot) : inputPath;
        outputPath = base + "_wt.wav";
    }

    // Read input
    WavData wav;
    if (!readWav(inputPath, wav)) return 1;

    std::cout << "input:  " << inputPath << " ("
              << wav.samples.size() << " samples, "
              << wav.sampleRate << " Hz)\n";
    std::cout << "mode:   " << mode << "\n";
    std::cout << "frames: " << numFrames << " x " << FRAME_SIZE << " samples\n";
    std::cout << "blend:  " << blendSamples << " samples\n";

    // Extract wavetable
    std::vector<float> output;
    if (mode == "spectral") {
        output = extractSpectral(wav, numFrames, blendSamples);
    } else {
        output = extractSlice(wav, numFrames, blendSamples);
    }

    // Write output
    if (!writeWav(outputPath, output, wav.sampleRate, numFrames, writeClm)) return 1;

    double seconds = (double)output.size() / wav.sampleRate;
    double sizeMB  = (double)(output.size() * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "output: " << outputPath << " ("
              << numFrames << " frames, "
              << std::fixed;
    std::cout.precision(1);
    std::cout << seconds << "s, " << sizeMB << " MB)\n";

    if (writeClm)
        std::cout << "serum:  clm marker included (auto-detected as wavetable)\n";

    return 0;
}
