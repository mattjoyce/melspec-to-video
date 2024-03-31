import argparse
import logging
import math
import os
import subprocess
import threading
import time
from typing import Any, Final, List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import psutil
import soundfile as sf
from params import Params
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import utils

# Mel Scale Spectrogram Video from Audio

# This script transforms audio files into visual spectrogram representations and encodes them into a video file.
# It leverages librosa for audio processing and Mel spectrogram generation, matplotlib for plotting, and FFmpeg for video encoding.

# Configuration is managed via a YAML file, allowing customization of video dimensions, frame rate, audio processing parameters, and optional color palettes for the spectrogram.
# The script supports dynamic adjustment of spectrogram image widths based on audio chunk sizes, ensuring smooth transitions and consistent visual output across the video.

# Features include:
# - Loading and processing of audio data in configurable chunks.
# - Generation of Mel spectrograms from audio data, with optional normalization against a global max power level.
# - Customizable spectrogram color mapping.
# - Efficient video encoding using FFmpeg, with parameters for quality and compatibility.

# Designed for flexibility and efficiency, this tool is suited for researchers, musicians, bioaucoustic field recordists, and audiovisual artists looking to create detailed and customized visualizations of audio data.


# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# these are used as globasl to track memory usgae
memory_usage = 0
max_mem = 0  # maximum allowed mem usage in %


def load_and_resample_mono(audio_path, target_sr=22050):
    # Load and resample audio file
    data, samplerate = sf.read(audio_path)

    # if there are multiple dimensions, it's not mono, avarage them
    if data.ndim > 1:
        logging.info(f"Converting to mono")
        data = np.mean(data, axis=1)  # Convert to mono

    data_resampled = librosa.resample(data, orig_sr=samplerate, target_sr=target_sr)
    return data_resampled, target_sr


def create_playhead_overlay(
    frame_number,
    frame_rate,
    image_size,
    playhead_position,
    line_color=(255, 0, 0, 128),
    line_width=2,
):
    """
    Create an overlay image with a semi-transparent playhead line.

    Parameters:
    - image_size: A tuple (width, height) specifying the size of the overlay.
    - playhead_position: A float representing the horizontal position of the playhead (0 to 1).
    - line_color: A tuple (R, G, B, A) specifying the color and opacity of the line.
    - line_width: The width of the line in pixels.

    Returns:
    - An Image object representing the overlay with the playhead line.
    """
    # Create a transparent overlay
    overlay = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Calculate the x position of the playhead line
    playhead_x = int(playhead_position * image_size[0])

    # Draw the semi-transparent playhead line on the overlay
    draw.line(
        [(playhead_x, 0), (playhead_x, image_size[1])],
        fill=line_color,
        width=line_width,
    )

    # Calculate the time at the playhead
    total_seconds = frame_number / frame_rate
    hours = math.floor(total_seconds / 3600)
    minutes = math.floor((total_seconds % 3600) / 60)
    seconds = math.floor(total_seconds % 60)
    tenths = int(
        (total_seconds - math.floor(total_seconds)) * 10
    )  # Get tenths of a second

    # Format the time mark as text
    time_mark = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{tenths}"

    # Draw the time mark text near the playhead line
    # Adjust the font size and position as needed
    font = ImageFont.load_default()
    text_position = (
        playhead_x + 5,
        10,
    )  # Position the text slightly to the right of the playhead line
    draw.text(text_position, time_mark, fill=line_color, font=font)

    return overlay


def monitor_memory_usage(interval=1):
    """Monitors memory usage at specified intervals (in seconds) and updates the global memory_usage variable."""
    global memory_usage, max_mem
    while True:
        memory = psutil.virtual_memory()
        memory_usage = (memory.used / memory.total) * 100
        if max_mem < memory_usage:
            max_mem = memory_usage
        time.sleep(interval)


def profile_audio(params: Params) -> dict[str, Any]:
    """Function to derive the global max power reference from an audio file."""
    # Initialization and configuration extraction
    audio_fsp = params["input"]
    logging.info(f"Audio file in: {audio_fsp}")
    target_sr = params["sr"]
    logging.info(f"Target Sample Rate: {target_sr}")

    # Configuration for Mel spectrogram
    melspec = params.get("mel_spectrogram", {})

    f_low = melspec.get("f_low", 0)
    f_high = melspec.get("f_high", 22050)
    hop_length = melspec.get("hop_length", 512)
    n_fft = melspec.get("n_fft", 2048)
    n_mels = melspec.get("n_mels", 100)

    print(f' f_low : { melspec["f_low"]}')
    print(f' f_high: { melspec["f_high"]}')

    # Load and resample the audio
    y, sr = load_and_resample_mono(audio_fsp, target_sr)
    profiling_chunk_duration = params.get("audio_visualization", {}).get(
        "profiling_chunk_duration", 60
    )
    samples_per_chunk = int(sr * profiling_chunk_duration)
    global_max_power = 0
    # Process each chunk
    for i in tqdm(range(0, len(y), samples_per_chunk)):
        y_chunk = y[i : i + samples_per_chunk]
        S = librosa.feature.melspectrogram(
            y=y_chunk,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=f_low,
            fmax=f_high,
        )
        maxpower = np.max(S)
        print(f"profiling chunk max power : {maxpower}")
        if maxpower > global_max_power:
            global_max_power = maxpower

    return {
        "max_power": float(global_max_power),
        "sample_count": len(y),
        "sample_rate": sr,
    }


def generate_spectrograms(
    params: Params,
    project: Params,
) -> bool:
    """Function processes the audio, creates a series of wide Mel spectrogram PNG files."""

    images_metadata = []  # each image will have an entry with it's metadata
    audio_metadata = project["audio_metadata"]
    max_power = audio_metadata.get("max_power")
    audio_fsp = params["input"]
    video = params.get("video", {})
    audiovis = params.get("audio_visualization", {})

    melspec = params.get("mel_spectrogram", {})
    f_low = melspec.get("f_low", 0)
    f_high = melspec.get("f_high", 22050)
    hop_length = melspec.get("hop_length", 512)
    n_fft = melspec.get("n_fft", 2048)
    n_mels = melspec.get("n_mels", 100)
    db_low = melspec.get("db_low", -70)
    db_high = melspec.get("db_high", 0)

    max_spectrogram_width = audiovis.get("max_spectrogram_width", 1000)
    logging.info(f"max_spectrogram_width: {max_spectrogram_width}")

    target_sample_rate = params["sr"]

    samples_per_pixel = (audiovis["seconds_in_view"] * target_sample_rate) / video[
        "width"
    ]
    y_chunk_samples = int(samples_per_pixel * max_spectrogram_width)
    logging.info(f"y_chunk_samples: {y_chunk_samples}")

    full_chunk_duration_secs = y_chunk_samples / target_sample_rate

    total_duration_secs = librosa.get_duration(path=audio_fsp, sr=target_sample_rate)

    current_position_secs = 0

    count = 0
    progress_bar = tqdm(total=total_duration_secs)
    while current_position_secs < total_duration_secs:
        # print(f"{current_position_secs} / {total_duration_secs}")

        duration_secs = min(
            y_chunk_samples / target_sample_rate,
            total_duration_secs - current_position_secs,
        )

        y, sr = librosa.load(
            audio_fsp,
            sr=target_sample_rate,
            offset=current_position_secs,
            duration=duration_secs,
        )

        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=f_low,
            fmax=f_high,
        )

        print(f"max power : { max_power }")
        print(f" db_low : { db_low }")
        print(f" db_high: { db_high }")
        print(f" f_low : { f_low }")
        print(f" f_high: { f_high }")

        S_dB = librosa.power_to_db(
            S,
            ref=max_power,
            amin=10 ** (db_low / 10.0),
            top_db=db_high - db_low,
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
            S_dB,
            sr=sr,
            cmap=audiovis.get("cmap", "magma"),
            hop_length=hop_length,
            fmin=f_low,
            fmax=f_high,
        )
        plt.axis("off")
        plt.tight_layout(pad=0)

        basename = os.path.splitext(audio_metadata["source_audio_filename"])[0]
        image_filename = f"{basename}-{count:04d}.png"
        image_path = os.path.join(project["project_path"], image_filename)

        if utils.allow_save(image_path, params.get("overwrite", None)):
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
    project["images_metadata"] = images_metadata
    project.save_to_json(os.path.join(project["project_path"], project["project_file"]))

    return True


def render_project_to_mp4(params: Params, project: Params) -> bool:
    """Function to produce MP$ video from a spectrogram

    This approach uses a sliding window crop to sample the spectrogram images, and feed to ffmpeg to encode.

    """

    # localise some variable we will use frequently
    video_fsp = os.path.join(project["project_path"], params["output"])
    print(video_fsp)

    # geometry of resulting mp4
    frame_width: Final[int] = params["video"].get("width", 800)
    frame_height: Final[int] = params["video"].get("height", 200)
    frame_rate: Final[int] = params["video"].get("frame_rate", 30)

    print(f"frame_height: {frame_height}")
    print(f"frame_width: {frame_width}")

    # duration_in_view: The amount of audio duration (in seconds) represented in a single frame of the video.
    # This determines how much of the audio waveform is 'in view' in the spectrogram for each frame,
    # affecting the visual pacing of the spectrogram scroll in the final video.
    seconds_in_view: Final[int] = params["audio_visualization"]["seconds_in_view"]

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

    print(ffmpeg_cmd)

    with open("ffmpeg_log.txt", "wb") as log_file:
        ffmpeg_process: subprocess.Popen = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT
        )

    ## Image Loop
    # Cycle through each image described by the project
    total_images = len(project.get("images_metadata", []))
    for i in range(total_images):
        image_metadata = project["images_metadata"][i]
        filename = image_metadata["filename"]
        print(filename)
        start_time = image_metadata["start_time"]
        end_time = image_metadata["end_time"]

        # duration of audio, the image represents
        image_duration = end_time - start_time
        print(f"image_duration : {image_duration}")

        # the number of frame we need to render
        num_frames = int(image_duration * frame_rate)
        print(f"num_frames : {num_frames}")

        # get the image
        current_image = Image.open(os.path.join(project["project_path"], filename))
        image_width, image_height = current_image.size

        print(f"image width : {image_width}")

        # number of pixels to slide for each frame
        step_px = image_width / num_frames
        print(f"step_px : {step_px}")

        # extend the image to cover the join
        if i < total_images - 1:
            next_image_metadata = project["images_metadata"][i + 1]
            next_filename = next_image_metadata["filename"]
            next_image = Image.open(
                os.path.join(project["project_path"], next_filename)
            )
            # Assume frame_width is the width of the frame to append from the next image
            next_image_section = next_image.crop((0, 0, frame_width, frame_height))
            current_image = concatenate_images(current_image, next_image_section)
        else:
            # Handle the last image separately if needed
            pass

        for i in range(num_frames):

            crop_start_x = int(i * step_px)
            crop_end_x = int(crop_start_x + frame_width)

            cropped_frame = current_image.crop(
                (crop_start_x, 0, crop_end_x, frame_height)
            )

            # Convert the image to bytes
            cropped_frame_rgb = cropped_frame.convert("RGB")
            cropped_frame_bytes = cropped_frame_rgb.tobytes()

            # Write the frame bytes to ffmpeg's stdin
            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.write(cropped_frame_bytes)

    # Close ffmpeg's stdin to signal end of input
    # ffmpeg_process.wait()
    if ffmpeg_process.stdin is not None:
        ffmpeg_process.stdin.close()

    return True


def concatenate_images(image1, image2):
    """Concatenate image2 to the right side of image1."""
    new_width = image1.width + image2.width
    new_img = Image.new("RGB", (new_width, image1.height))
    new_img.paste(image1, (0, 0))
    new_img.paste(image2, (image1.width, 0))
    return new_img


def main():

    # Start the memory monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_memory_usage, args=(1,), daemon=True
    )
    monitor_thread.start()

    parser = argparse.ArgumentParser(
        description="Generate a scrolling spectrogram video from an audio file."
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path."
    )

    parser.add_argument(
        "--start",
        help="Audio start time in seconds. Default: 0.",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--duration",
        help="Audio clip duration in seconds. Processes till end if unspecified.",
        type=float,
    )

    parser.add_argument(
        "--sr",
        help="Audio sample rate. Overrides the source sample rate.",
        type=int,
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

    parser.add_argument(
        "-d",
        "--dated",
        action="store_true",
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
    )

    args = parser.parse_args()
    params = Params(args.config, args=args, file_type="yaml")
    print(params)

    full_path = os.path.abspath(args.input)
    directory_path = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    basename, extension = os.path.splitext(filename)

    print(f"The full path of the input file is: {full_path}")

    project_folder = utils.generate_project_folder_name(
        basename, params.get("dated", None)
    )

    if not os.path.exists(project_folder):
        utils.create_folder(directory_path, project_folder)
    else:
        print("project folder exists")

    if params.get("overwrite", None):
        basename = os.path.split(params["input"])[0]
        print(f"Deleting project media from {project_folder}")
        utils.clear_project_media(project_folder, basename)

    print(f"Project folder : {project_folder}")

    json_filename = os.path.join(project_folder, "project.json")

    if os.path.exists(json_filename):
        print("Using existing project file")
        project = Params(file_path=json_filename, file_type="json")
    else:
        default_project_structure = {
            "project_path": project_folder,
            "project_file": "project.json",
            "audio_metadata": {
                "source_audio_path": "",
                "source_audio_filename": "",
                "max_power": None,
                "sample_rate": None,
                "sample_count": None,
            },
            "images_metadata": [],
        }
        project = Params(default_config=default_project_structure)

    project["audio_metadata"]["source_audio_path"] = directory_path
    project["audio_metadata"]["source_audio_filename"] = args.input

    maxpower = project["audio_metadata"].get("max_power")
    sample_rate = project["audio_metadata"].get("sample_rate")
    sample_count = project["audio_metadata"].get("sample_count")
    if any(value is None for value in [maxpower, sample_rate, sample_count]):
        # This block executes if any of maxpower, sample_rate, or sample_count is None
        profile = profile_audio(params)  #
        print(f"Profile : {profile}")
        project["audio_metadata"].update(profile)

    project.save_to_json(json_filename)
    print(f"project data : {project}")
    ### PASS 2
    # Generate the spectrograms
    generate_spectrograms(params, project)

    render_project_to_mp4(params, project)

    # process_audio(config, args)


if __name__ == "__main__":
    main()
