# Ambisonic Workflow for Zoom H3-VR Recordings

## Overview

Your Zoom H3-VR records in **A-Format** by default - raw signals from 4 tetrahedral microphone capsules. For melspec-to-video processing, you need to convert to **B-Format** (W, X, Y, Z spherical harmonics) first.

## File Format Summary

- **A-Format**: Raw capsule signals (what you have)
- **B-Format (AmbiX)**: W, X, Y, Z spherical harmonics (what we need)
- **B-Format (FuMa)**: Legacy format with different normalization

## Workflow Options

### Option 1: Use Python Converter (Recommended)

Use the provided `aformat_to_bformat.py` script:

```bash
# Single file conversion
python3.11 aformat_to_bformat.py input.WAV output_ambix.wav

# Batch convert all files in directory
python3.11 aformat_to_bformat.py ~/Documents/ZoomH3VR/ --batch
```

Batch mode will:
- Convert all `*.WAV` files in the directory
- Output to `converted/` subdirectory
- Add `_AmbiX` suffix to filenames

### Option 2: Use Zoom Ambisonics Player

Manual conversion through the Zoom player GUI:
1. Open file in Zoom Ambisonics Player
2. Select "AmbiX" format
3. Export each file individually

**Pros**: Uses Zoom's exact proprietary conversion matrix
**Cons**: Manual, one-by-one process

## Using B-Format Files with melspec_to_video.py

Once converted to B-Format, you can extract mono audio for spectrograms:

### Method 1: Extract W Channel Only (Recommended)

The W channel is the omnidirectional component - perfect for mono:

```python
# In melspec_to_video.py, modify librosa.load() calls:

# Instead of:
y, _ = librosa.load(path=input_path, sr=target_sr, offset=start, duration=duration, mono=True)

# Use:
y, _ = librosa.load(path=input_path, sr=target_sr, offset=start, duration=duration, mono=False)
y = y[0]  # Extract W channel (omnidirectional)
```

This needs changing in:
- Line 1085-1087 (profile command)
- Line 1291-1293 (spectrograms command)

### Method 2: Pre-extract W Channel to Stereo/Mono File

Use ffmpeg to extract just the W channel:

```bash
# Extract W channel (channel 0) as mono
ffmpeg -i input_ambix.wav -map_channel 0.0.0 output_mono.wav

# Or as stereo (duplicate W to both channels)
ffmpeg -i input_ambix.wav -map_channel 0.0.0 -map_channel 0.0.0 output_stereo.wav
```

Then use these mono/stereo files with your existing pipeline.

## Verifying Conversion Quality

To check if the Python converter matches Zoom's output:

1. Convert one file with both methods
2. Compare W channels:

```bash
# Extract W from Python-converted file
ffmpeg -i python_converted.wav -map_channel 0.0.0 python_w.wav

# Extract W from Zoom-converted file
ffmpeg -i zoom_converted.wav -map_channel 0.0.0 zoom_w.wav

# Listen and compare
```

If they sound significantly different, the conversion matrix may need adjustment.

## Matrix Adjustment (Advanced)

If the standard matrix doesn't match Zoom's output, you can tweak coefficients in `aformat_to_bformat.py`:

1. Set `matrix_type="custom"`
2. Modify the custom matrix coefficients (around line 68)
3. Test and compare until it matches

## Why Not Average All 4 Channels?

**Don't** average A-Format or B-Format channels:
- A-Format: Channels are raw capsule signals with random phases
- B-Format: X, Y, Z are bipolar (positive/negative) directional components
- Averaging causes unpredictable phase cancellation and artifacts

**Always use W channel only** for mono reference.

## Batch Processing Example

Complete workflow for your ZoomH3VR directory:

```bash
cd ~/Projects/melspec-to-video

# 1. Batch convert A→B Format
python3.11 aformat_to_bformat.py ~/Documents/ZoomH3VR/ --batch --pattern "251105_*.WAV"

# 2. Extract W channels as mono for quick processing
mkdir ~/Documents/ZoomH3VR/mono
for f in ~/Documents/ZoomH3VR/converted/*_AmbiX.wav; do
    basename=$(basename "$f" _AmbiX.wav)
    ffmpeg -i "$f" -map_channel 0.0.0 ~/Documents/ZoomH3VR/mono/${basename}_W.wav
done

# 3. Use W channel files with melspec_to_video.py
python melspec_to_video.py --config config.yaml profile \
    --input ~/Documents/ZoomH3VR/mono/251105_001_W.wav \
    --path ./output
```

## Future Enhancements

Potential ideas for spatial visualization:
- 4-channel spectrogram (W, X, Y, Z in 2x2 grid)
- Rotating perspective decode (orbit around soundfield)
- Directional energy visualization (color-coded by direction)
- Circular/polar spectrogram showing azimuth

See the main discussion for details.
