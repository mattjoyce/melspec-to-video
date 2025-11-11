# Ambisonic Integration Complete

## What Was Implemented

The `melspec_to_video.py` pipeline now automatically detects and handles ambisonic files:

### Automatic Format Detection

When loading audio, the system:

1. **Checks channel count** - Must be 4-channel for first-order ambisonics
2. **Reads WAV metadata** using ffprobe to detect format:
   - `Rec Mode=Ambisonics A` → A-Format
   - `Rec Mode=AmbiX` → B-Format (AmbiX)
   - Channel labels (`zTRK1=W` vs `zTRK1=FLU`) as fallback

### Automatic Handling

**A-Format files** (raw capsule signals):
- Automatically converted to B-Format using standard tetrahedral matrix
- W channel extracted for mono output
- Conversion happens on-the-fly during audio loading

**B-Format/AmbiX files** (W, Y, Z, X):
- W channel (channel 0) extracted directly
- No conversion needed

**Regular files** (mono/stereo):
- Processed normally using librosa's default mono conversion
- Fully backward compatible

## Testing Results

✓ **B-Format detection**: Successfully detected `251105_001_Ambix.wav`
✓ **W channel extraction**: Extracted omnidirectional channel from AmbiX
✓ **A-Format detection**: Successfully detected `251105_001.WAV`
✓ **A→B conversion**: Converted and extracted W channel

## Usage

No changes needed to your workflow! Just use ambisonic files directly:

```bash
# Works with A-Format, B-Format, or regular audio
python3.11 melspec_to_video.py --config config-final-48k.yaml profile \
    --input ~/Documents/ZoomH3VR/251105_001_Ambix.wav \
    --path ./output

python3.11 melspec_to_video.py --config config-final-48k.yaml spectrograms \
    --path ./output

python3.11 melspec_to_video.py --config config-final-48k.yaml mp4 \
    --path ./output
```

The pipeline automatically:
- Detects the format
- Converts if needed (A→B)
- Extracts W channel for mono
- Generates spectrograms and video

## Technical Details

### Functions Added

**`detect_ambisonic_format(file_path)`**
- Returns 'A', 'B', or None
- Checks metadata using ffprobe
- Looks for Zoom-specific markers

**`aformat_to_bformat_simple(capsules)`**
- Standard tetrahedral conversion matrix
- Returns AmbiX order: W, Y, Z, X

**`load_audio_with_ambisonic_support(file_path, sr, offset, duration)`**
- Replaces `librosa.load()` calls
- Automatic detection and handling
- Returns mono audio data

### Modified Locations

- Line ~1225: `profile` command audio loading
- Line ~1433: `spectrograms` command audio loading

## Channel Order Reference

**AmbiX (Zoom's B-Format)**:
- Channel 0: W (omnidirectional) ← **Used for mono**
- Channel 1: Y (left-right)
- Channel 2: Z (up-down)
- Channel 3: X (front-back)

**A-Format (Zoom's raw recording)**:
- Channel 0-3: FLU, FRD, BLD, BRU (capsule signals)
- Requires conversion to B-Format

## Why W Channel?

The W channel in B-Format is the **omnidirectional pressure component** - it represents the complete sound field without directional encoding. This is the natural mono representation of ambisonic audio.

**Don't** average all channels:
- In A-Format: Averaging causes unpredictable phase artifacts
- In B-Format: Y, Z, X are bipolar (±) directional components
- Only W is purely positive omnidirectional

## Conversion Quality

The standard tetrahedral conversion matrix is used:
```python
W = 0.5 * (A + B + C + D)   # Omnidirectional
X = 0.5 * (A + B - C - D)   # Front-Back
Y = 0.5 * (A - B + C - D)   # Left-Right
Z = 0.5 * (A - B - C + D)   # Up-Down
```

This produces results very close to Zoom's proprietary conversion. If higher fidelity is needed, you can:
1. Use Zoom Ambisonics Player to batch convert A→B
2. Adjust the matrix coefficients in `aformat_to_bformat_simple()`

## Future Enhancements

Potential spatial visualization features:
- 4-channel spectrogram grid (W, Y, Z, X)
- Rotating perspective decode (orbiting the sound field)
- Directional energy visualization (color-coded by azimuth)
- Circular/polar spectrogram layout

These would leverage the full spatial information rather than just using W.

## Dependencies Added

- `soundfile` - For reading WAV metadata
- `subprocess` (built-in) - For ffprobe metadata extraction

All other dependencies were already present.
