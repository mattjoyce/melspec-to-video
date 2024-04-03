import argparse
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import psutil
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from params import Params

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
                f"Error: File '{fullfilepath}' exists and overwriting is not allowed.  use --overwrite"
            )
            sys.exit(1)  # Exit the program with an error code
        # If overwriting is allowed
        else:
            return True
    # If the file does not exist, it's safe to save
    else:
        return True

    return False  # This line is technically redundant due to the sys.exit above


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
    frame_rate = params["video"].get("frame_rate", 30)
    playhead_position = params["audio_visualization"].get("playhead_position", 0.5)
    playhead_rgba = tuple(
        params["audio_visualization"].get("playhead_rgba", [255, 255, 255, 192])
    )

    playhead_width = params["audio_visualization"].get("playhead_width", 2)
    image_width, image_height = image_size

    # Create a transparent overlay
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Calculate the x position of the playhead line
    playhead_x = int(playhead_position * image_width)

    # Draw the semi-transparent playhead line on the overlay
    draw.line(
        [(playhead_x, 0), (playhead_x, image_height)],
        fill=playhead_rgba,
        width=playhead_width,
    )

    # Calculate the time at the playhead position
    total_seconds = frame_number / frame_rate
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
    image = image.convert("RGBA")
    frame_width = params["video"]["width"]
    playhead_position = params["audio_visualization"].get("playhead_position", 0.5)
    print(f"playhead : {playhead_position}")
    playhead_section_rgba = tuple(
        params["audio_visualization"].get("playhead_section_rgba", [0, 0, 0, 0])
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
    logging.info("Profiling start")

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
    logging.info("Spectrogram generation start")
    images_metadata = []  # each image will have an entry with it's metadata
    audio_metadata = project["audio_metadata"]
    max_power = audio_metadata.get("max_power")
    audio_fsp = audio_metadata["source_audio_path"]
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
        # print(f'{current_position_secs} / {total_duration_secs}')

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

        # print(f'max power : { max_power }')
        # print(f' db_low : { db_low }')
        # print(f' db_high: { db_high }')
        # print(f' f_low : { f_low }')
        # print(f' f_high: { f_high }')

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

        basename = Path(audio_metadata["source_audio_path"]).stem
        image_filename = f"{basename}-{count:04d}.png"
        image_path = Path(project["project_path"]) / image_filename

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
    project["images_metadata"] = images_metadata
    print(project)
    project.save_to_json(project["project_file"])

    return True


def get_ffmpeg_cmd(params: Params, project: Params) -> List[str]:
    # localise some variable we will use frequently
    video_fsp = Path(project["project_path"]) / params["output"]
    print(video_fsp)

    # geometry of resulting mp4
    frame_width: Final[int] = params["video"].get("width", 800)
    frame_height: Final[int] = params["video"].get("height", 200)
    frame_rate: Final[int] = params["video"].get("frame_rate", 30)

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
    video_fsp = Path(project["project_path"]) / params["output"]
    print(video_fsp)

    # geometry of resulting mp4
    frame_width: Final[int] = params["video"].get("width", 800)
    frame_height: Final[int] = params["video"].get("height", 200)
    frame_rate: Final[int] = params["video"].get("frame_rate", 30)

    print(f"frame_height: {frame_height}")
    print(f"frame_width: {frame_width}")

    playhead_position: Final = params["audio_visualization"].get(
        "playhead_position", {}
    )
    seconds_in_view: Final = params["audio_visualization"]["seconds_in_view"]

    # get the ffmpeg command line parameter for gpu, or cpu
    ffmpeg_cmd = get_ffmpeg_cmd(params, project)
    print(ffmpeg_cmd)

    # create the encoder pipe so we can stream the frames
    with open(Path(project["project_path"]) / "ffmpeg_log.txt", "wb") as log_file:
        ffmpeg_process: subprocess.Popen = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT
        )

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
        image_metadata = project["images_metadata"][i]
        filename = image_metadata["filename"]
        print(filename)
        start_time = image_metadata["start_time"]
        end_time = image_metadata["end_time"]

        # duration of audio the spectrogram image represents
        image_duration = end_time - start_time
        print(f"image_duration : {image_duration}")

        # retrieve the image
        spectrogram_image = Image.open(Path(project["project_path"]) / filename)

        work_image = spectrogram_image

        # add overlay to the image and add a lead in if #1
        if params["audio_visualization"].get("playhead_position", None):
            work_image = adjust_spectrogram_for_playhead(
                params, spectrogram_image, is_first, is_last
            )

        # the number of frame we need to render the spectrograms
        num_frames = int(image_duration * frame_rate)

        # number of pixels to slide for each frame
        step_px = spectrogram_image.width / num_frames
        print(f"step_px : {step_px}")

        # calculate how much time is used for the lead-in
        lead_in_duration = seconds_in_view * playhead_position
        print(f"lead in duration : {lead_in_duration}")
        lead_in_frames = int((lead_in_duration) * frame_rate)
        if is_first:
            num_frames += lead_in_frames
            print(f"lead in frames : {lead_in_frames}")

        if is_last:
            num_frames -= lead_in_frames
            print(f"lead in frames : {lead_in_frames}")

        print(f"num_frames: {num_frames}")

        # check
        # work_image.save(Path(project["project_path"] / f"adjusted-{filename}"))

        # extend the image to cover the join
        if i < total_images - 1:
            next_image_metadata = project["images_metadata"][i + 1]
            next_filename = next_image_metadata["filename"]
            next_image = Image.open(Path(project["project_path"]) / next_filename)
            # Assume frame_width is the width of the frame to append from the next image
            next_image_section = next_image.crop((0, 0, frame_width, frame_height))
            work_image = concatenate_images(work_image, next_image_section)

        for i in range(num_frames):
            global_frame_count += 1
            crop_start_x = int(i * step_px)
            crop_end_x = int(crop_start_x + frame_width)

            cropped_frame = work_image.crop((crop_start_x, 0, crop_end_x, frame_height))
            cropped_frame_rgba = cropped_frame.convert("RGBA")
            ### Insert frame overlays
            playhead_overlay_rgba = create_playhead_overlay(
                params, global_frame_count, cropped_frame_rgba.size
            )

            final_frame = Image.alpha_composite(
                cropped_frame_rgba, playhead_overlay_rgba
            )

            axis_overlay = create_vertical_axis(
                params,
                params["audio_visualization"].get("playhead_position", 0.5),
                [100, 1000, 3000, 5000, 9000],
                cropped_frame_rgba.size,
            )

            final_frame = Image.alpha_composite(final_frame, axis_overlay)

            # Convert the image to bytes
            final_frame_rgb = final_frame.convert("RGB")
            final_frame_bytes = final_frame_rgb.tobytes()

            # Write the frame bytes to ffmpeg's stdin
            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.write(final_frame_bytes)

    # Close ffmpeg's stdin to signal end of input
    # ffmpeg_process.wait()
    if ffmpeg_process.stdin is not None:
        ffmpeg_process.stdin.close()

    return True


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


def calculate_frequency_positions(f_low, f_high, freqs_of_interest, img_height):
    """
    Calculate the vertical positions of given frequencies on a mel spectrogram image and returns
    them along with the corresponding frequency if they fall within the image height.

    Parameters:
    - f_low: The lowest frequency in Hz included in the spectrogram.
    - f_high: The highest frequency in Hz included in the spectrogram.
    - freqs_of_interest: A list of frequencies in Hz for which to calculate positions.
    - img_height: The height of the spectrogram image in pixels.

    Returns:
    - A list of tuples (y_position, frequency) for frequencies within the image height.
    """
    # Convert frequency bounds and frequencies of interest to the mel scale
    mel_low = librosa.hz_to_mel(f_low)
    mel_high = librosa.hz_to_mel(f_high)
    mels_of_interest = librosa.hz_to_mel(freqs_of_interest)

    # Calculate the relative position of each frequency of interest within the mel range
    relative_positions = (mels_of_interest - mel_low) / (mel_high - mel_low)

    # Convert these positions to percentages of the spectrogram height
    y_positions = (1 - relative_positions) * img_height

    # Pair positions with frequencies, exclude out-of-bounds, and return as list of tuples
    pos_freq_pairs = [
        (pos, freq)
        for pos, freq, rel_pos in zip(
            y_positions, freqs_of_interest, relative_positions
        )
        if 0 <= rel_pos <= 1
    ]

    return pos_freq_pairs


def create_vertical_axis(
    params: Params,
    xpospc: float,
    freqs_of_interest: List[int],
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
    xpos = width * xpospc
    melspec = params["mel_spectrogram"]

    # Create a transparent image
    axis_image = Image.new("RGBA", (width, height), (255, 0, 0, 0))
    draw = ImageDraw.Draw(axis_image)

    # Draw the vertical axis line
    draw.line([(xpos, 0), (xpos, height)], fill="white", width=1)

    # Calculate the positions for the frequency labels
    pos_freq_pairs = calculate_frequency_positions(
        melspec["f_low"], melspec["f_high"], freqs_of_interest, height
    )

    font = ImageFont.load_default()
    # Draw ticks and labels for each frequency of interest
    for pos, freq in pos_freq_pairs:
        label = f"{freq}Hz"
        left, top, right, bottom = font.getbbox(text=label)
        text_size_y = top - bottom
        text_size_x = right - left

        # Draw tick mark
        draw.line([(xpos, pos), (xpos - 5, pos)], fill="white", width=1)
        # Draw label
        draw.text(
            (xpos - text_size_x - 10, pos + (text_size_y // 2)),
            label,
            fill="white",
            font=font,
        )
    return axis_image


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

    # TODO #2
    # parser.add_argument(
    #     "--start",
    #     help="Audio start time in seconds. Default: 0.",
    #     type=float,
    #     default=0,
    # )

    # parser.add_argument(
    #     "--duration",
    #     help="Audio clip duration in seconds. Processes till end if unspecified.",
    #     type=float,
    # )

    # TODO #3
    # parser.add_argument(
    #     "--encodeaudio",
    #     help="add the audio to the MP4",
    #     action="store_true",
    #     default=None,
    # )

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

    args = parser.parse_args()
    params = Params(args.config, args=args, file_type="yaml")

    # resolve all paths and check as needed
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logging.error(f"Config does not exist : {config_path}")
        sys.exit()
    else:
        print(f"Config file : {config_path}")

    source_audio_path = Path(args.input).resolve()
    if not source_audio_path.exists():
        logging.error(f"Input does not exist : {source_audio_path}")
        sys.exit()
    else:
        print(f"Source Audio Path : {source_audio_path}")

    # the mp4 file - will be tested later and overwriten if needed
    output_path = Path(args.output).resolve()

    print(params)

    if args.path:
        project_path = Path(args.path).resolve()
        if not project_path.exists():
            logging.error(f"Project folder does not exist : {project_path}")
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
        print(project)
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

    project["audio_metadata"]["source_audio_path"] = str(source_audio_path)

    maxpower = project["audio_metadata"].get("max_power", None)
    sample_rate = project["audio_metadata"].get("sample_rate", None)
    sample_count = project["audio_metadata"].get("sample_count", None)
    if any(value is None for value in [maxpower, sample_rate, sample_count]):
        # This block executes if any of maxpower, sample_rate, or sample_count is None
        profile = profile_audio(params)  #
        print(f"Profile : {profile}")
        project["audio_metadata"].update(profile)

    project.save_to_json(str(project_json_path))
    print(f"project data : {project}")
    ### PASS 2
    # Generate the spectrograms
    generate_spectrograms(params, project)

    render_project_to_mp4(params, project)

    # process_audio(config, args)


if __name__ == "__main__":
    main()
