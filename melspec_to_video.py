""" Module to analyse audio, and make videos from spectrograms."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Final, List, Tuple, cast, Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from params import Params

# Mel Scale Spectrogram Video from Audio

# This script transforms audio files into visual spectrogram representations
# and encodes them into a video file.
# It leverages librosa for audio processing and Mel spectrogram generation,
# matplotlib for plotting, and FFmpeg for video encoding.

# Configuration is managed via a YAML file, allowing customization of video
# dimensions, frame rate, audio processing parameters, and optional color
# palettes for the spectrogram.
# The script supports dynamic adjustment of spectrogram image widths #
# based on audio chunk sizes, ensuring smooth transitions and
# consistent visual output across the video.

# Features include:
# - Loading and processing of audio data in configurable chunks.
# - Generation of Mel spectrograms from audio data, with optional normalization
#   against a global max power level.
# - Customizable spectrogram color mapping.
# - Efficient video encoding using FFmpeg, with parameters for quality and compatibility.

# Designed for flexibility and efficiency
# this tool is suited for researchers, musicians, bioaucoustic field recordists,
# and audiovisual artists looking to create detailed and customized visualizations of audio data.


# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_color(color_value: Any) -> tuple[int, int, int, int]:
    """
    Converts a color value to a tuple of integers (RGBA format).

    Args:
    color_value (Any): The color information, expected to be a list of integers
                       in RGBA format, but can be any type due to dynamic data sources.

    Returns:
    tuple[int, int, int, int]: The color as an RGBA tuple.
    """
    # Ensure the color_value is a list of integers; provide a default if not
    if isinstance(color_value, List) and all(
        isinstance(item, int) for item in color_value
    ):
        rgba_color = tuple(color_value)
    else:
        # Default color (white with some transparency)
        rgba_color = (255, 255, 255, 192)

    # Use cast to inform mypy about the expected return type explicitly
    return cast(tuple[int, int, int, int], rgba_color)


def allow_save(fullfilepath: Path, allowoverwrite: bool) -> bool:
    """
    Determines if a file can be saved based on its existence and the overwrite policy.

    Args:
        fullfilepath (str): The full path to the file to save.
        allowoverwrite (bool): Whether overwriting an existing file is allowed.

    Returns:
        bool: True if the file can be saved, False otherwise.
    """
    # Check if the file exists
    if fullfilepath.exists():
        # If overwriting is not allowed, print an error and exit
        if not allowoverwrite:
            print(
                f"Error: File '{fullfilepath}' exists and overwriting is not allowed."
            )
            print("use --overwrite")
            sys.exit(1)  # Exit the program with an error code
        # If overwriting is allowed
        else:
            return True
    # If the file does not exist, it's safe to save
    else:
        return True

    return False  # This line is technically redundant due to the sys.exit above


def load_and_resample_mono(
    params: Params,
    start_secs: Optional[float] = float(0),
    duration_secs: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load a segment of an audio file, resample it to a specified sample rate, and ensure it is mono.

    This function uses the librosa library to load and resample audio. It allows for specifying
    a start time and duration for loading only a part of the audio file, which is useful for large files
    or specific analysis of audio segments.

    Parameters:
    - params (Params): Configuration parameters containing the audio file path, sample rate,
                       start time, and duration for the segment to be loaded and resampled.

    Returns:
    - Tuple[np.ndarray, int]: A tuple containing the audio data as a numpy array and the sample rate as an integer.
    """

    # Extract necessary parameters
    if not start_secs:
        start_secs = params.get("start_time", 0)
        print(f"Start not specificied, using cli or 0 : {start_secs}")

    if not duration_secs:
        duration_secs = params.get("duration", None)
        print(f"Duration not specificied, using cli or None: {duration_secs}")

    # Determine the total duration of the audio file if duration is not explicitly provided
    if duration_secs is None:
        total_duration = librosa.get_duration(path=params.input)
        duration_secs = total_duration - float(start_secs)
        # Ensure that the calculated duration is not negative
        duration_secs = max(duration_secs, 0)
        print(f"Duration calculated: {duration_secs}")

    # Load and resample the specified audio segment
    y, sr = librosa.load(
        path=params.input,
        sr=params.sr,
        offset=start_secs,
        duration=duration_secs,
        mono=True,
    )

    return y, int(sr)


def create_playhead_overlay(
    params: Params, frame_number: int, image_size: Tuple[int, int]
):
    """
    Create an overlay image with a semi-transparent playhead line indicating the current playback position.

    Args:
        params (Params): A Params object containing video and audio visualization configurations.
        frame_number (int): The current frame number in the video sequence.
        image_size (Tuple[int, int]): The size of the overlay image (width, height).

    Returns:
        Image: An RGBA Image object representing the overlay with the semi-transparent playhead line.
    """
    playhead = params.overlays["playhead"]

    # convert list to pillow color
    playhead_rgba = get_color(playhead.get("playhead_rgba", [255, 255, 255, 192]))

    playhead_width = playhead.get("playhead_width", 2)
    image_width, image_height = image_size

    # Create a transparent overlay
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Calculate the x position of the playhead line
    playhead_x = int(playhead.get("playhead_position", 0.5) * image_width)

    # Draw the semi-transparent playhead line on the overlay
    draw.line(
        [(playhead_x, 0), (playhead_x, image_height)],
        fill=playhead_rgba,
        width=playhead_width,
    )

    # Calculate the time at the playhead position
    total_seconds = frame_number / params.video.get("frame_rate", 30)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the time mark as text
    time_mark = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds - int(seconds)) * 10)}"

    # Draw the time mark text near the playhead line
    font = ImageFont.load_default()
    text_position = (playhead_x + 5, 10)  # Adjust text position as needed
    draw.text(text_position, time_mark, fill=playhead_rgba, font=font)

    return overlay


def adjust_spectrogram_for_playhead(
    params: Params, image: Image.Image, is_first: bool, is_last: bool
) -> Image.Image:
    """function to extend the spectrogram to accomodate moving the playhead"""
    image = image.convert("RGBA")
    frame_width = params.video["width"]
    playhead = params.overlays["playhead"]
    playhead_position = playhead.get("playhead_position", 0.5)
    print(f"playhead : {playhead_position}")
    playhead_section_rgba = get_color(
        playhead.get("playhead_section_rgba", [0, 0, 0, 0])
    )
    if is_first:
        lead_in_width = int(frame_width * playhead_position)
        lead_in_section = Image.new(
            "RGBA", (lead_in_width, image.height), playhead_section_rgba
        )
        print(f"lead in size : {lead_in_section.size}")
        image = concatenate_images(lead_in_section, image)

    if is_last:
        play_out_width = int(frame_width * (1 - playhead_position))
        play_out_section = Image.new(
            "RGBA", (play_out_width, image.height), playhead_section_rgba
        )
        print(f"play out size : {play_out_section.size}")
        image = concatenate_images(image, play_out_section)

    return image


def profile_audio(params: Params) -> dict[str, Any]:
    """Function to derive the global max power reference from an audio file."""
    # Initialization and configuration extraction
    logging.info("PROFILING START")

    logging.info("Audio file in: %s", params.input)

    if not params.sr:
        params.sr = librosa.get_samplerate(params.input)

    logging.info("Target Sample Rate: %s", params.sr)

    # Configuration for Mel spectrogram
    melspec = params.get("mel_spectrogram", {})

    print(f' f_low : { melspec["f_low"]}')
    print(f' f_high: { melspec["f_high"]}')

    profiling_chunk_duration = params.get("audio_visualization", {}).get(
        "profiling_chunk_duration", 60
    )

    y, _ = load_and_resample_mono(params)

    samples_per_chunk = int(params.sr * profiling_chunk_duration)
    global_max_power = 0
    # Process each chunk
    for i in tqdm(range(0, len(y), samples_per_chunk)):
        y_chunk = y[i : i + samples_per_chunk]
        s = librosa.feature.melspectrogram(
            y=y_chunk,
            sr=params.sr,
            n_fft=melspec.get("n_fft", 2048),
            hop_length=melspec.get("hop_length", 512),
            n_mels=melspec.get("n_mels", 100),
            fmin=melspec.get("f_low", 0),
            fmax=melspec.get("f_high", params.sr / 2),
        )
        maxpower = np.max(s)
        # print(f"profiling chunk max power : {maxpower}")
        global_max_power = max(global_max_power, maxpower)

    return {
        "max_power": float(global_max_power),
        "sample_count": len(y),
        "sample_rate": params.sr,
    }


def generate_spectrograms(
    params: Params,
    project: Params,
) -> bool:
    """Function processes the audio, creates a series of wide Mel spectrogram PNG files."""
    logging.info("Spectrogram generation start")
    images_metadata = []  # each image will have an entry with it's metadata
    audio_metadata = project.audio_metadata
    video = params.get("video", {})
    audiovis = params.get("audio_visualization", {})

    melspec = params.get("mel_spectrogram", {})

    max_spectrogram_width = audiovis.get("max_spectrogram_width", 1000)
    logging.info("max_spectrogram_width: %s", max_spectrogram_width)

    # Directly assign the target_sample_rate with a fallback to the project's sample_rate or 22050
    target_sample_rate = params.get(
        "sr", project.audio_metadata.get("sample_rate", 22050)
    )

    samples_per_pixel = (audiovis["seconds_in_view"] * target_sample_rate) / video[
        "width"
    ]
    y_chunk_samples = int(samples_per_pixel * max_spectrogram_width)
    logging.info("y_chunk_samples: %d ", y_chunk_samples)

    full_chunk_duration_secs = y_chunk_samples / target_sample_rate
    start_time = params.get("start", 0)  # Safely get start_time with a default of 0
    print(f"start_time : {start_time}")
    duration = params.get(
        "duration", None
    )  # Safely get duration with a default of None
    # If duration is not provided (None), calculate it
    if duration is None:
        # Calculate the remaining duration of the audio file from start_time
        audio_file_duration = librosa.get_duration(
            path=audio_metadata["source_audio_path"], sr=target_sample_rate
        )
        print(f"audio_file_duration : {audio_file_duration}")
        duration = audio_file_duration - start_time
        # Ensure calculated duration is not negative
        duration = max(duration, 0)
    print(f"duration : {duration}")

    # Adjusted to consider the effective processing duration
    total_duration_secs = duration
    print(f"total_duration_secs : {total_duration_secs}")

    current_position_secs = 0

    count = 0
    progress_bar = tqdm(total=total_duration_secs)
    while current_position_secs < total_duration_secs:
        # print(f'{current_position_secs} / {total_duration_secs}')

        # the last chunk can be samller so...
        duration_secs = min(
            y_chunk_samples / target_sample_rate,
            total_duration_secs - current_position_secs,
        )

        y, sr = load_and_resample_mono(params, current_position_secs, duration_secs)

        s = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=melspec.get("n_fft", 2048),
            hop_length=melspec.get("hop_length", 512),
            n_mels=melspec.get("n_mels", 100),
            fmin=melspec.get("f_low", 0),
            fmax=melspec.get("f_high", params.sr / 2),
        )

        s_db = librosa.power_to_db(
            s,
            ref=audio_metadata.get("max_power", np.max(s)),
            amin=10 ** (melspec.get("db_low", -70) / 10.0),
            top_db=melspec.get("db_high", 0) - melspec.get("db_low", -70),
        )

        if (
            duration_secs < full_chunk_duration_secs
        ):  # Adjust for potentially shorter last chunk
            image_width = int(
                max_spectrogram_width * (duration_secs / full_chunk_duration_secs)
            )
        else:
            image_width = max_spectrogram_width

        plt.figure(figsize=(image_width / 100, video.get("height", 100) / 100))
        librosa.display.specshow(
            s_db,
            sr=sr,
            cmap=audiovis.get("cmap", "magma"),
            hop_length=melspec.get("hop_length", 512),
            fmin=melspec.get("f_low", 0),
            fmax=melspec.get("f_high", params.sr / 2),
        )
        plt.axis("off")
        plt.tight_layout(pad=0)

        basename = Path(audio_metadata["source_audio_path"]).stem
        image_filename = f"{basename}-{count:04d}.png"
        image_path = Path(project.project_path) / image_filename

        if allow_save(image_path, params.get("overwrite", None)):
            plt.savefig(
                image_path,
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close()

        # Save spectrogram and update project data with image metadata
        # For each generated image, create its metadata entry
        image_metadata = {
            "filename": image_filename,
            "start_time": current_position_secs,
            "end_time": current_position_secs + duration_secs,
            # Additional metadata can be included here
        }
        # Append this metadata to the project data
        # Consider creating a function similar to update_project_data to handle this update
        images_metadata.append(image_metadata)

        current_position_secs += duration_secs
        progress_bar.update(duration_secs)
        count += 1
    progress_bar.close()
    project.images_metadata = images_metadata
    print(project)
    project.save_to_json(project.project_file)

    return True


def get_ffmpeg_cmd(params: Params, project: Params) -> List[str]:
    """Function to select the best ffmpeg command line arguments"""

    # localise some variable we will use frequently
    video_fsp = Path(project.project_path) / params.output
    print(video_fsp)

    # geometry of resulting mp4
    frame_width: Final[int] = params.video.get("width", 800)
    frame_height: Final[int] = params.video.get("height", 200)
    frame_rate: Final[int] = params.video.get("frame_rate", 30)

    ffmpeg_cmd_cpu: List[str] = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{frame_width}x{frame_height}",  # Set frame size
        "-pix_fmt",
        "rgb24",  # Set pixel format
        "-r",
        f"{frame_rate}",
        "-i",
        "-",  # Input from stdin
        "-c:v",
        "libx264",  # Use libx264 for H.264 encoding
        "-crf",
        "17",  # Adjust CRF as needed for balance between quality and file size
        "-preset",
        "fast",  # Preset for encoding speed/quality trade-off (options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        "-vf",
        "format=yuv420p",  # Pixel format for compatibility
        f"{video_fsp}",
    ]

    ### Start the ffmpeg process
    ffmpeg_cmd_gpu: List[str] = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{frame_width}x{frame_height}",  # Set frame size
        "-pix_fmt",
        "rgb24",  # Set pixel format
        "-r",
        f"{frame_rate}",
        "-i",
        "-",  # Input from stdin
        "-c:v",
        "h264_nvenc",  # nvidia acceleration
        "-crf",
        "17",  # Constant rate factor for quality
        "-preset",
        "slow",  # Preset for encoding speed
        "-vf",
        "format=yuv420p",  # Pixel format for compatibility
        f"{video_fsp}",
    ]

    ffmpeg_cmd = ffmpeg_cmd_gpu
    if params.get("cpu", None):
        ffmpeg_cmd = ffmpeg_cmd_cpu

    return ffmpeg_cmd


def render_project_to_mp4(params: Params, project: Params) -> bool:
    """Function to produce MP$ video from a spectrogram

    This approach uses a sliding window crop to sample the spectrogram images, and feed to ffmpeg to encode.

    """
    logging.info("MP4 Encoding start")
    # localise some variable we will use frequently
    video_fsp = Path(project.project_path) / params.output
    print(video_fsp)

    # geometry of resulting mp4
    frame_width: Final[int] = params.video.get("width", 800)
    frame_height: Final[int] = params.video.get("height", 200)
    frame_rate: Final[int] = params.video.get("frame_rate", 30)

    print(f"frame_height: {frame_height}")
    print(f"frame_width: {frame_width}")

    seconds_in_view: Final = params.audio_visualization["seconds_in_view"]

    # get the ffmpeg command line parameter for gpu, or cpu
    print(get_ffmpeg_cmd(params, project))

    # create the encoder pipe so we can stream the frames
    with open(Path(project.project_path) / "ffmpeg_log.txt", "wb") as log_file:
        with subprocess.Popen(
            get_ffmpeg_cmd(params, project),
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        ) as ffmpeg_process:

            ## Image Loop
            # Cycle through each image described by the project

            # counthe images in the metadata
            total_images = len(project.get("images_metadata", []))

            # setup a global counmt, we use this to calculate time in overlay
            global_frame_count = 0

            # step through each spectrograph image
            for i in range(total_images):
                # first and last may need treatment
                is_first = True if i == 0 else False
                is_last = True if i == total_images - 1 else False

                print(f"Image number {i+1} of {total_images}")
                image_metadata = project.images_metadata[i]
                filename = image_metadata["filename"]
                print(filename)

                # duration of audio the spectrogram image represents
                image_duration = (
                    image_metadata["end_time"] - image_metadata["start_time"]
                )
                print(f"image_duration : {image_duration}")

                # retrieve the image
                spectrogram_image: Final = Image.open(
                    Path(project.project_path) / filename
                )

                # the number of frames we need to render the spectrograms
                num_frames = int(image_duration * frame_rate)

                # number of pixels to slide for each frame
                step_px = spectrogram_image.width / num_frames
                print(f"step_px : {step_px}")
                work_image = spectrogram_image

                # if playhead is enabled, the image size changes
                # this alters the number of frames needed to render
                # actural overlay is added later
                if params.overlays["playhead"].get("enabled", None):
                    playhead_position = params.overlays["playhead"].get(
                        "playhead_position", 0
                    )
                    work_image = adjust_spectrogram_for_playhead(
                        params, spectrogram_image, is_first, is_last
                    )

                    # calculate how much time is used for the lead-in
                    lead_in_duration = seconds_in_view * playhead_position
                    print(f"lead in duration : {lead_in_duration}")
                    lead_in_frames = int((lead_in_duration) * frame_rate)
                    if is_first:
                        num_frames += lead_in_frames

                    if is_last:
                        num_frames -= lead_in_frames

                print(f"num_frames: {num_frames}")

                # check
                # work_image.save(Path(project["project_path"] / f"adjusted-{filename}"))

                # as we are using a sliding window crop, at the end of the image, we spillover
                # into the next image, so we just append one whole frame width from the next
                if i < total_images - 1:
                    next_image_metadata = project.images_metadata[i + 1]
                    next_image = Image.open(
                        Path(project.project_path) / next_image_metadata["filename"]
                    )
                    # Assume frame_width is the width of the frame to append from the next image
                    next_image_section = next_image.crop(
                        (0, 0, frame_width, frame_height)
                    )
                    work_image = concatenate_images(work_image, next_image_section)

                # cropping, streaming, encoding loop
                for i in range(num_frames):
                    global_frame_count += 1
                    crop_start_x = round(i * step_px)
                    crop_end_x = round(crop_start_x + frame_width)

                    cropped_frame = work_image.crop(
                        (crop_start_x, 0, crop_end_x, frame_height)
                    )
                    cropped_frame_rgba = cropped_frame.convert("RGBA")

                    cropped_frame_rgba = apply_overlays(
                        params, global_frame_count, cropped_frame_rgba
                    )

                    # Convert the image to bytes
                    final_frame_rgb = cropped_frame_rgba.convert("RGB")
                    final_frame_bytes = final_frame_rgb.tobytes()

                    # Write the frame bytes to ffmpeg's stdin
                    ffmpeg_process.stdin.write(final_frame_bytes)

    return True


def apply_overlays(params: Params, global_frame_count: int, cropped_frame_rgba: Image):
    """apply the overlays to the cropped frame and return resulting image"""
    if params.overlays["playhead"].get("enabled", None):
        # create overlay
        playhead_overlay_rgba = create_playhead_overlay(
            params, global_frame_count, cropped_frame_rgba.size
        )
        # apply to frame
        cropped_frame_rgba = Image.alpha_composite(
            cropped_frame_rgba, playhead_overlay_rgba
        )

    if params.overlays.get("labels", {}).get("enabled", None):
        # create overlay
        label_overlay = create_labels_overlay(
            params,
            global_frame_count,
            cropped_frame_rgba.size,
        )
        # apply to frame
        cropped_frame_rgba = Image.alpha_composite(cropped_frame_rgba, label_overlay)

    if params.overlays["frequency_axis"].get("enabled", None):
        # create overlay
        axis_overlay = create_vertical_axis_overlay(
            params,
            cropped_frame_rgba.size,
        )
        # apply to frame
        cropped_frame_rgba = Image.alpha_composite(cropped_frame_rgba, axis_overlay)

    return cropped_frame_rgba


def concatenate_images(image1, image2):
    """Concatenate image2 to the right side of image1, ensuring both are RGBA."""
    # Ensure both images are in RGBA mode
    image1 = image1.convert("RGBA")
    image2 = image2.convert("RGBA")

    new_width = image1.width + image2.width
    new_img = Image.new("RGBA", (new_width, image1.height))
    new_img.paste(image1, (0, 0), image1)
    new_img.paste(image2, (image1.width, 0), image2)
    return new_img


def calculate_frequency_positions(
    f_low: float, f_high: float, freqs_of_interest: List[float], img_height: int
) -> List[Tuple[float, float]]:
    """
    Calculate the vertical positions of given frequencies on a mel spectrogram image. Frequencies
    outside the specified range are mapped to either 0 or the maximum image height.

    Parameters:
    - f_low: The lowest frequency (in Hz) included in the spectrogram.
    - f_high: The highest frequency (in Hz) included in the spectrogram.
    - freqs_of_interest: A list of frequencies (in Hz) for which to calculate positions.
    - img_height: The height of the spectrogram image in pixels.

    Returns:
    - A list of tuples (y_position, frequency) for all frequencies. Frequencies below f_low
      are assigned a y_position of 0, and frequencies above f_high are assigned a y_position of img_height.
    """
    # Convert the low and high frequency bounds, as well as the frequencies of interest, into the mel scale.
    mel_low = librosa.hz_to_mel(f_low)
    mel_high = librosa.hz_to_mel(f_high)
    mels_of_interest = librosa.hz_to_mel(freqs_of_interest)

    # Calculate the relative position of each frequency of interest within the total mel range.
    relative_positions = (mels_of_interest - mel_low) / (mel_high - mel_low)

    # Convert these relative positions to actual y-axis positions on the spectrogram image.
    y_positions = (1 - relative_positions) * img_height

    # Pair each calculated position with its corresponding frequency.
    # Adjust positions for out-of-range frequencies to either 0 (top) or img_height (bottom).
    pos_freq_pairs = [
        (max(0, min(pos, img_height)), freq)  # Clamp position within [0, img_height]
        for pos, freq in zip(y_positions, freqs_of_interest)
    ]

    return pos_freq_pairs


def create_vertical_axis_overlay(
    params: Params,
    image_size,
):
    """
    Generates a transparent image with a vertical frequency axis.

    Parameters:
    - xpos: The x-position (in pixels) for the vertical axis line.
    - f_low, f_high: Frequency range (in Hz) for the axis.
    - freqs_of_interest: A list of frequencies (in Hz) where ticks and labels will be placed.
    - img_height, image_width: Dimensions of the output image.

    Returns:
    - A PIL Image object with the specified vertical frequency axis.
    """

    # localise some variables
    width, height = image_size
    axis = params.overlays.get("frequency_axis")
    x_pos = width * axis["axis_position"]
    ink_color = get_color(axis["axis_rgba"])
    melspec = params.mel_spectrogram

    # Create a transparent image
    axis_image = Image.new("RGBA", (width, height), (255, 0, 0, 0))
    draw = ImageDraw.Draw(axis_image)

    # Draw the vertical axis line
    draw.line([(x_pos, 0), (x_pos, height)], fill=ink_color, width=1)

    # Calculate the positions for the frequency labels
    pos_freq_pairs = calculate_frequency_positions(
        melspec["f_low"], melspec["f_high"], axis["freq_hz"], height
    )

    font = ImageFont.load_default()
    # Draw ticks and labels for each frequency of interest
    for pos, freq in pos_freq_pairs:
        label = f"{freq}Hz"
        left, top, right, bottom = font.getbbox(text=label)
        text_size_y = top - bottom
        text_size_x = right - left

        # Draw tick mark
        draw.line([(x_pos, pos), (x_pos - 5, pos)], fill=ink_color, width=1)
        # Draw label
        draw.text(
            (x_pos - text_size_x - 10, pos + (text_size_y // 2)),
            label,
            fill=ink_color,
            font=font,
        )
    return axis_image


def save_config(params: Params) -> bool:
    # setup some keys we want to exclude
    params.set_exclusions(["config", "saveconfig"])
    print(params.exclusions)
    newfile = Path(params.saveconfig).resolve()
    params.save_to_yaml(str(newfile))
    return True


def create_labels_overlay(
    params: Params,
    global_frame_index: int,
    frame_size: Tuple[int, int],
) -> Image:
    """
    Creates an overlay of labels on a transparent image based on the spectrogram parameters and frame data.

    This function calculates the position and size of labels based on the current frame's time window and
    draws them onto a transparent image to be overlayed onto the video frame.

    Args:
        params (Params): Configuration and parameters for the spectrogram.
        project (Params): Project specific configuration, including audio metadata.
        global_frame_index (int): The index of the current frame in the video.
        frame_size (Tuple[int, int]): The width and height of the frame.

    Returns:
        Image: A PIL Image object with the drawn labels.
    """

    # Create a transparent image for labels
    label_image = Image.new("RGBA", frame_size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(label_image)

    # Calculate the duration of one pixel in the frame
    time_per_pixel = params.audio_visualization["seconds_in_view"] / label_image.width

    # Calculate the start and end time of the current frame
    frame_start_secs = global_frame_index / params.video["frame_rate"]
    frame_end_secs = frame_start_secs + params.audio_visualization["seconds_in_view"]

    for label in params.overlays["labels"]["items"]:

        # if time not specified assume all time
        x_pos_ratio = label.get("x_pos_ratio", None)
        time_range = label.get("time", [frame_start_secs, frame_end_secs])

        frequency_range = label["freq"]
        ink_color = get_color(label["rgba"])  # Ensure ink_color is a tuple

        label_start_secs, label_end_secs = time_range

        # Check if the label is within the current frame's time range
        if frame_start_secs <= label_start_secs <= label_end_secs <= frame_end_secs:
            # Calculate label positions
            y_px_list = calculate_frequency_positions(
                params.mel_spectrogram["f_low"],
                params.mel_spectrogram["f_high"],
                freqs_of_interest=frequency_range,
                img_height=label_image.height,
            )

            # Convert time to pixel positions
            if x_pos_ratio:
                x0 = x1 = x_pos_ratio * label_image.width
            else:
                x0 = round((label_start_secs - frame_start_secs) / time_per_pixel)
                x1 = round((label_end_secs - frame_start_secs) / time_per_pixel)
            y0 = max(float(y_px_list[1][0]), 0)
            y1 = min(float(y_px_list[0][0]), label_image.height)

            text_x, text_y = x0, y0

            if label["text"] == "trafic":
                print(x0, y0)

            # Draw label based on its type
            match label["type"]:
                case "box":
                    draw.rectangle((x0, y0, x1, y1), outline=ink_color, width=1)
                case "point":
                    draw.ellipse(
                        [x0, y0, x0 + 7.5, y0 + 7.5],
                        outline=ink_color,
                        width=1,
                        fill=ink_color,
                    )
                case "brace":
                    draw.line((x0, y0, x0, y1), fill=ink_color, width=1)  # vertical
                    draw.line(
                        (x0, y0, x0 + 10, y0), fill=ink_color, width=1
                    )  # tick top
                    draw.line(
                        (x0, y1, x0 + 10, y1), fill=ink_color, width=1
                    )  # tick bottom

            # Drawing text
            font = ImageFont.load_default()
            left, top, right, bottom = font.getbbox(text=label["text"])
            text_size_y = top - bottom
            text_size_x = right - left

            v_align = label.get("v_align", None)
            match v_align:
                case "top":
                    text_y = y0
                case "bottom":
                    text_y = y1 - text_size_y
                case _:  # default to middle of freq raange
                    text_y = y0 + ((y1 - y0) / 2) + (text_size_y / 2)

            draw.text(
                (text_x - text_size_x - 10, text_y),
                label["text"],
                fill=ink_color,
                font=font,
            )
    return label_image


def main():
    """process command line argument and config"""
    parser = argparse.ArgumentParser(
        description="Generate a scrolling spectrogram video from an audio file."
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path."
    )

    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument(
        "--duration", type=float, default=None, help="Duration to process in seconds"
    )

    parser.add_argument(
        "--sr",
        help="Audio sample rate. Overrides the source sample rate.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--cpu",
        help="use CPU for processing if GPU is not available.",
        action="store_true",
        default=None,
    )

    parser.add_argument("-in", "--input", required=True, help="Input audio file path.")

    parser.add_argument(
        "-out",
        "--output",
        required=False,
        help='Output video file path. Default: "output.mp4".',
        default="output.mp4",
    )

    # we default bool args to None, so they are falsy but not == False
    # if it was False the defautl behavioir or argparse, it would always overwrite config yaml

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to project folder, if ommited, current folder will be used.",
        default=None,
    )

    parser.add_argument(
        "--saveconfig",
        help="Combine args with config and save.",
        default=None,
    )

    args = parser.parse_args()
    params = Params(args.config, args=args, file_type="yaml")

    # resolve all paths and check as needed
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logging.error("Config does not exist : %s", config_path)
        sys.exit()
    else:
        print(f"Config file : {config_path}")

    source_audio_path = Path(args.input).resolve()
    if not source_audio_path.exists():
        logging.error("Input does not exist : %s", source_audio_path)
        sys.exit()
    else:
        print(f"Source Audio Path : {source_audio_path}")

    if args.saveconfig:
        save_config(params)
    print(params)

    ## Project Data
    if args.path:
        project_path = Path(args.path).resolve()
        if not project_path.exists():
            logging.error("Project folder does not exist : %s", project_path)
            sys.exit()
    else:
        project_path = Path.cwd().resolve()
    print(f"Project path : {project_path}")

    project_json_path = Path(project_path) / "project.json"
    print(f"Project file : {project_json_path}")
    print(project_json_path)

    if project_json_path.exists():
        print("Using existing project file")
        project = Params(file_path=project_json_path, file_type="json")
    else:
        default_project_structure = {
            "project_path": str(project_path),
            "project_file": str(project_json_path),
            "audio_metadata": {
                "source_audio_path": str(source_audio_path),
                "max_power": None,
                "sample_rate": None,
                "sample_count": None,
            },
            "images_metadata": [],
        }
        project = Params(default_config=default_project_structure)

    project.audio_metadata["source_audio_path"] = str(source_audio_path)

    maxpower = project.audio_metadata.get("max_power", None)
    sample_rate = project.audio_metadata.get("sample_rate", None)
    sample_count = project.audio_metadata.get("sample_count", None)
    if (
        any(value is None for value in [maxpower, sample_rate, sample_count])
        or args.overwrite
    ):
        # This block executes if any of maxpower, sample_rate, or sample_count is None
        profile = profile_audio(params)  #
        print(f"Profile : {profile}")
        project.audio_metadata.update(profile)

    project.save_to_json(str(project_json_path))
    print(f"project data : {project}")

    generate_spectrograms(params, project)

    render_project_to_mp4(params, project)


if __name__ == "__main__":
    main()
