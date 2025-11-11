"""
A-Format to B-Format Converter for Zoom H3-VR recordings

This script converts A-Format (raw tetrahedral capsule) recordings to
B-Format AmbiX (W, X, Y, Z spherical harmonics).

Standard tetrahedral conversion matrix is used. If results don't match
Zoom's converter exactly, matrix coefficients can be adjusted.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def aformat_to_bformat(
    capsules: np.ndarray,
    matrix_type: str = "standard"
) -> np.ndarray:
    """
    Convert A-Format (4 capsule signals) to B-Format (W, X, Y, Z).

    Args:
        capsules: numpy array of shape (samples, 4) containing A-Format signals
        matrix_type: conversion matrix to use
            - 'standard': Standard tetrahedral matrix
            - 'fuma': FuMa-normalized output
            - 'custom': Use custom matrix (modify in code)

    Returns:
        numpy array of shape (samples, 4) containing B-Format signals (W, X, Y, Z)
    """
    # Ensure input shape
    if capsules.shape[1] != 4:
        raise ValueError(f"Expected 4 channels, got {capsules.shape[1]}")

    # Extract individual capsule signals
    # Note: Actual channel mapping for Zoom H3-VR may differ
    # You may need to experiment with different orderings
    A = capsules[:, 0]  # Capsule 1
    B = capsules[:, 1]  # Capsule 2
    C = capsules[:, 2]  # Capsule 3
    D = capsules[:, 3]  # Capsule 4

    if matrix_type == "standard":
        # Standard tetrahedral array conversion
        # Based on tetrahedral geometry with capsules at:
        # vertices of a regular tetrahedron
        W = 0.5 * (A + B + C + D)           # Omnidirectional
        X = 0.5 * (A + B - C - D)           # Front-Back
        Y = 0.5 * (A - B + C - D)           # Left-Right
        Z = 0.5 * (A - B - C + D)           # Up-Down

    elif matrix_type == "fuma":
        # FuMa normalization (different scaling)
        sqrt2 = np.sqrt(2)
        W = (A + B + C + D) / (2 * sqrt2)
        X = (A + B - C - D) / 2
        Y = (A - B + C - D) / 2
        Z = (A - B - C + D) / 2

    elif matrix_type == "custom":
        # Custom matrix - adjust these coefficients to match Zoom's output
        # You can determine correct values by comparing with Zoom player output
        W = 0.5 * (A + B + C + D)
        X = 0.5 * (A + B - C - D)
        Y = 0.5 * (A - B + C - D)
        Z = 0.5 * (A - B - C + D)

    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    # Stack channels into B-Format array (W, X, Y, Z)
    bformat = np.stack([W, X, Y, Z], axis=1)

    return bformat


def convert_file(
    input_path: Path,
    output_path: Path,
    matrix_type: str = "standard",
    output_format: str = "ambix"
) -> bool:
    """
    Convert a single A-Format file to B-Format.

    Args:
        input_path: Path to A-Format WAV file
        output_path: Path to output B-Format WAV file
        matrix_type: Conversion matrix to use
        output_format: 'ambix' or 'fuma'

    Returns:
        True if successful
    """
    try:
        # Read A-Format file
        logging.info(f"Reading: {input_path.name}")
        aformat, samplerate = sf.read(str(input_path))  # Convert Path to string

        # Verify it's 4-channel
        if aformat.ndim == 1 or aformat.shape[1] != 4:
            logging.error(f"File {input_path.name} is not 4-channel (A-Format)")
            return False

        # Convert to B-Format
        logging.info(f"Converting with '{matrix_type}' matrix...")
        bformat = aformat_to_bformat(aformat, matrix_type)

        # Write B-Format file
        logging.info(f"Writing: {output_path.name}")
        sf.write(
            str(output_path),  # Convert Path to string
            bformat,
            samplerate,
            subtype='PCM_24'  # Match Zoom's 24-bit output
        )

        logging.info(f"✓ Converted successfully")
        return True

    except Exception as e:
        logging.error(f"Error converting {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.WAV",
    matrix_type: str = "standard",
    suffix: str = "_AmbiX"
) -> Tuple[int, int]:
    """
    Batch convert all A-Format files in a directory.

    Args:
        input_dir: Directory containing A-Format files
        output_dir: Directory for B-Format output files
        pattern: File pattern to match (default: "*.WAV")
        matrix_type: Conversion matrix to use
        suffix: Suffix to add to output filenames

    Returns:
        Tuple of (successful_count, total_count)
    """
    # Find all matching files
    input_files = sorted(input_dir.glob(pattern))

    if not input_files:
        logging.warning(f"No files matching '{pattern}' found in {input_dir}")
        return 0, 0

    logging.info(f"Found {len(input_files)} files to convert")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each file
    successful = 0
    for input_path in tqdm(input_files, desc="Converting"):
        # Generate output filename
        output_filename = f"{input_path.stem}{suffix}.wav"
        output_path = output_dir / output_filename

        if convert_file(input_path, output_path, matrix_type):
            successful += 1

    return successful, len(input_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Zoom H3-VR A-Format recordings to B-Format (AmbiX)"
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input A-Format WAV file or directory"
    )

    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output B-Format WAV file or directory (optional for batch mode)"
    )

    parser.add_argument(
        "--matrix",
        choices=["standard", "fuma", "custom"],
        default="standard",
        help="Conversion matrix type (default: standard)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all WAV files in input directory"
    )

    parser.add_argument(
        "--pattern",
        default="*.WAV",
        help="File pattern for batch mode (default: *.WAV)"
    )

    parser.add_argument(
        "--suffix",
        default="_AmbiX",
        help="Suffix for output filenames in batch mode (default: _AmbiX)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logging.error(f"Input path does not exist: {args.input}")
        return 1

    # Single file mode
    if args.input.is_file() and not args.batch:
        if not args.output:
            # Generate output filename
            args.output = args.input.parent / f"{args.input.stem}_AmbiX.wav"

        success = convert_file(args.input, args.output, args.matrix)
        return 0 if success else 1

    # Batch mode
    elif args.input.is_dir() or args.batch:
        input_dir = args.input if args.input.is_dir() else args.input.parent
        output_dir = args.output if args.output else input_dir / "converted"

        successful, total = batch_convert(
            input_dir,
            output_dir,
            args.pattern,
            args.matrix,
            args.suffix
        )

        logging.info(f"\nConversion complete: {successful}/{total} files successful")
        return 0 if successful == total else 1

    else:
        logging.error("Invalid input/output combination")
        return 1


if __name__ == "__main__":
    exit(main())
