#!/usr/bin/env python3.11
"""
Extract W channel (omnidirectional) from B-Format ambisonic files.

This creates mono WAV files containing only the W channel, which is
perfect for spectrogramvisualization and general mono playback.
"""

import argparse
import logging
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_w_channel(input_path: Path, output_path: Path) -> bool:
    """Extract W channel (channel 0) from B-Format file."""
    try:
        # Read B-Format file
        logging.info(f"Reading: {input_path.name}")
        bformat, samplerate = sf.read(str(input_path))

        # Check if it's multi-channel
        if bformat.ndim == 1:
            logging.warning(f"File is already mono, copying as-is")
            w_channel = bformat
        else:
            # Extract W channel (first channel)
            w_channel = bformat[:, 0]
            logging.info(f"Extracted W channel from {bformat.shape[1]}-channel file")

        # Write mono file
        logging.info(f"Writing: {output_path.name}")
        sf.write(
            str(output_path),
            w_channel,
            samplerate,
            subtype='PCM_24'
        )

        logging.info(f"✓ Extracted successfully")
        return True

    except Exception as e:
        logging.error(f"Error extracting from {input_path.name}: {e}")
        return False


def batch_extract(input_dir: Path, output_dir: Path, pattern: str = "*.wav"):
    """Batch extract W channel from all files in directory."""
    input_files = sorted(input_dir.glob(pattern))

    if not input_files:
        logging.warning(f"No files matching '{pattern}' found in {input_dir}")
        return 0, 0

    logging.info(f"Found {len(input_files)} files to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    successful = 0
    for input_path in tqdm(input_files, desc="Extracting W channel"):
        output_filename = f"{input_path.stem}_W.wav"
        output_path = output_dir / output_filename

        if extract_w_channel(input_path, output_path):
            successful += 1

    return successful, len(input_files)


def main():
    parser = argparse.ArgumentParser(
        description="Extract W channel (omnidirectional) from B-Format files"
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input B-Format WAV file or directory"
    )

    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output mono WAV file or directory"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all files in directory"
    )

    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="File pattern for batch mode (default: *.wav)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logging.error(f"Input does not exist: {args.input}")
        return 1

    # Single file mode
    if args.input.is_file() and not args.batch:
        if not args.output:
            args.output = args.input.parent / f"{args.input.stem}_W.wav"

        success = extract_w_channel(args.input, args.output)
        return 0 if success else 1

    # Batch mode
    elif args.input.is_dir() or args.batch:
        input_dir = args.input if args.input.is_dir() else args.input.parent
        output_dir = args.output if args.output else input_dir / "mono"

        successful, total = batch_extract(input_dir, output_dir, args.pattern)

        logging.info(f"\nExtraction complete: {successful}/{total} files successful")
        return 0 if successful == total else 1

    else:
        logging.error("Invalid input/output combination")
        return 1


if __name__ == "__main__":
    exit(main())
