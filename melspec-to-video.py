import subprocess
import time
import math

import logging
import argparse


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

import threading
import psutil

from typing import Final
"""
Mel Scale Spectrogram Video from Audio

This script transforms audio files into visual spectrogram representations and encodes them into a video file. 
It leverages librosa for audio processing and Mel spectrogram generation, matplotlib for plotting, and FFmpeg for video encoding.

Configuration is managed via a YAML file, allowing customization of video dimensions, frame rate, audio processing parameters, and optional color palettes for the spectrogram. 
The script supports dynamic adjustment of spectrogram image widths based on audio chunk sizes, ensuring smooth transitions and consistent visual output across the video.

Features include:
- Loading and processing of audio data in configurable chunks.
- Generation of Mel spectrograms from audio data, with optional normalization against a global max power level.
- Customizable spectrogram color mapping.
- Efficient video encoding using FFmpeg, with parameters for quality and compatibility.

Designed for flexibility and efficiency, this tool is suited for researchers, musicians, bioaucoustic field recordists, and audiovisual artists looking to create detailed and customized visualizations of audio data.
"""

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# these are used as globasl to track memory usgae
memory_usage = 0
max_mem = 0  # maximum allowed mem usage in %


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


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def calculate_buffer(source_audio_duration, time_per_frame, mel_buffer_multiplier, buffering):
    # if not buffering, read the whole thing, set the buffer to be big enough
    if buffering:
        # the ammount of audio data needed to produce the wide mel buffer
        audio_buffer = time_per_frame * mel_buffer_multiplier
        # we add enough to service the sliding window tail
        extended_audio_buffer = audio_buffer + time_per_frame
    else:
        # make the buffer as big as the audio
        audio_buffer = source_audio_duration
        extended_audio_buffer = source_audio_duration + time_per_frame
    return audio_buffer, extended_audio_buffer


def tune_buffer(mel_buffer_multiplier, frame_width, width_limit):
    global max_mem
    # Calculate the maximum buffer multiplier based on the image width limit
    max_mel_buffer_multiplier = int(width_limit / frame_width) - 2

    # Adjust the buffer size based on memory usage
    if max_mem < 90:
        adjusted_multiplier = int(mel_buffer_multiplier * 1.5)
    elif max_mem > 90:  # Memory usage is high, reduce buffer size
        adjusted_multiplier = int(mel_buffer_multiplier * 0.8)
    else:
        adjusted_multiplier = mel_buffer_multiplier

    # Cap the adjusted_multiplier at the max_mel_buffer_multiplier
    adjusted_multiplier = min(adjusted_multiplier, max_mel_buffer_multiplier)

    # Log the adjusted (and possibly capped) buffer size
    if adjusted_multiplier != mel_buffer_multiplier:
        logging.info(f"Adjusted buffer to {adjusted_multiplier}")
    else:
        logging.info("Buffer size remains unchanged.")

    # Reset max memory usage after adjustment
    max_mem = 0

    return adjusted_multiplier


def is_max_mel_image_width_safe(source_audio_duration, time_per_frame, frame_width, limit):
    # we need to know the most audio we might handle
    max_audio_duration = source_audio_duration + time_per_frame

    # Determine the maximum number of frames needed
    max_num_frames = (
        math.ceil(max_audio_duration / time_per_frame) + 1
    )  # Round up to ensure all audio is covered, include an extra frame for sliding window

    # Calculate the maximum possible wide Mel image width
    max_wide_mel_image_width = max_num_frames * frame_width

    is_safe = max_wide_mel_image_width <= limit
    logging.info(f"limit : {limit}")
    logging.info(f"max_wide_mel_image_width: {max_wide_mel_image_width}")
    return is_safe, max_wide_mel_image_width  # is safe?


def process_audio(config, args):
    #these are used to track memory usage and provide warnings
    global max_mem, memory_usage

    # extract the config and argumates to variables.
    audio_fsp = args.input
    video_fsp = args.output
    logging.info(f"Audio file in : {audio_fsp}")
    logging.info(f"Video file out: {video_fsp}")

    sr: Final = args.sr
    logging.info(f"Audio Sample Rate : {sr}")

    audio_start = args.start
    audio_duration = args.duration

    # TODO
    """ logging.info(f"Audio start offset (secs) : {audio_start}")
        if audio_duration:
            logging.info(f"Audio clip duration (secs) : {audio_duration}") """

    if args.maxpower:
        maxpower = args.maxpower
    else:
        maxpower = config.get("mel_spectrogram", {}).get("maxpower", None)

    ##loads the file
    source_audio_duration : Final = librosa.get_duration(path=audio_fsp)
    logging.info(f"Total duration (secs): {source_audio_duration}")

    logging.info(f"Resample rate : {sr} hz")
    logging.info(f"Upper Reference Power : {maxpower}")

    # get the output video paramters from config
    frame_width = config.get("video", {}).get("width", 800)
    frame_height = config.get("video", {}).get("height", 200)
    frame_rate = config.get("video", {}).get("frame_rate", 30)
    logging.info(
        f"Output Video : {frame_width}px x {frame_height}px @ {frame_rate}fps "
    )

    # get the cofig used to create the mel scale spectrograms
    #   the color pallate
    #   https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colormap = config.get("audio_visualization", {}).get("cmap", "magma")
    if colormap not in plt.colormaps():
        logging.warning("'Colormap not found, using 'magma'")
        colormap = "magma"

    logging.info(f"Spectrogram Pallette : {colormap} ")

    # config for the spectrogram
    #   time_per_frame, is the duration of aution represented in 1 x frame_width
    time_per_frame = config["audio_visualization"]["time_per_frame"]


    # get the playhead position from config.
    # 0 is hard left of the frame, 0.5 is centre
    playhead = config.get("audio_visualization", {}).get("playhead_position", 0.0)
    lead_in_silence_duration = time_per_frame * playhead
    tail_silence_duration = time_per_frame * (1 - playhead)


    # Maximum allowable image width
    max_image_width_px = 65536  # Safe limit, matplot or png

    is_safe_size, calculated_size = is_max_mel_image_width_safe(
        source_audio_duration, time_per_frame, frame_width, max_image_width_px
    )

    # Check if the maximum possible width exceeds the maximum allowable width
    if not is_safe_size and not args.buffering:
        error_message = (
            f"Error: The maximum possible width of the wide Mel image ({calculated_size} pixels) "
            f"exceeds the maximum allowable width ({max_image_width_px} pixels). "
            "Please consider reducing the audio length or using buffering."
        )
        logging.error(error_message)
        raise ValueError(error_message)

    #   a multiplyer  that determines how much audio to process each cycle
    #   smaller numbers will take longer, higher numbers will use a lot of memory
    #   1 minute displayed in a frame x 20 = 20 minutes of audio processed
    mel_buffer_multiplier = config["audio_visualization"]["mel_buffer_multiplier"]

    audio_buffer, extended_audio_buffer = calculate_buffer(
        source_audio_duration, time_per_frame, mel_buffer_multiplier, args.buffering
    )

    logging.info(f"Full frame duration : {time_per_frame} secs")
    logging.info(f"Audio Buffer     : {audio_buffer} secs")
    logging.info(f"Audio Buffer ext : {extended_audio_buffer} secs")

    # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
    f_low = config.get("mel_spectrogram", {}).get("f_low", 0)
    f_high = config.get("mel_spectrogram", {}).get("f_high", None)
    hop_length = config.get("mel_spectrogram", {}).get("hop_length", 512)
    n_ftt = config.get("mel_spectrogram", {}).get("n_fft", 2048)
    db_low = config.get("mel_spectrogram", {}).get("db_low", 60)
    db_high = config.get("mel_spectrogram", {}).get("db_high", 0)

    logging.info(f"Frequency range : {f_low} to {f_high} Hz")
    logging.info(f"db range : {db_low} to {db_high} dB")
    logging.info(f"n_fft : {n_ftt}")
    logging.info(f"hop length : {hop_length}")

    ffmpeg_cmd_cpu = [
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
    ffmpeg_cmd_gpu = [
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
    if args.cpu:
        ffmpeg_cmd = ffmpeg_cmd_cpu

    with open("ffmpeg_log.txt", "wb") as log_file:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT
        )

    ##### start work
    current_position_secs = 0
    total_frames_rendered = 0

    # the while loop is focussed on the actual audio durations, not silence padding
    while current_position_secs < source_audio_duration:
        logging.info(f"Starting Chunk")
        logging.critical(f"current_position = {current_position_secs}")
        ## check if this is the last chunk
        is_last_chunk = (current_position_secs + audio_buffer) >= source_audio_duration

        if is_last_chunk:
            # Calculate the duration of tail silence based on the playhead position
            tail_silence_duration = time_per_frame * (1 - playhead)

            # Adjust remaining_duration to include the actual audio left plus the tail silence
            remaining_duration = source_audio_duration - current_position_secs + tail_silence_duration

            # Since this is the last chunk, the audio buffer needs to accommodate the remaining audio plus the tail silence
            audio_buffer = remaining_duration

            # No next chunk to interleave with, so extended_audio_buffer is set equal to audio_buffer
            extended_audio_buffer = audio_buffer

            # Convert the duration of silence into a corresponding image width
            # This assumes a linear relationship between time and image width as used in the rest of the video
            silence_image_width = int(frame_width * (tail_silence_duration / time_per_frame))

            # Now adjust the calculation of wide_mel_image_width for the last chunk to include the visual silence
            proportion_of_full_chunk = remaining_duration / (time_per_frame * mel_buffer_multiplier)
            wide_mel_image_width = int((frame_width * mel_buffer_multiplier) * proportion_of_full_chunk)

            # Append the visual silence to the wide Mel image width
            wide_mel_image_width += silence_image_width


        # Load audio segment
        logging.info("Loading audio segment")
        start_time = time.time()
        y, sr = librosa.load(
            audio_fsp,
            sr=sr,
            offset=current_position_secs,
            duration=extended_audio_buffer,
        )
        logging.info(f"Processing time: {(time.time() - start_time):.2f} seconds")

        # prepend an offset to position the readhead to centre
        # first chunk only
        # playhead positioning
        if current_position_secs == 0:
            silence = np.zeros(int(sr * lead_in_silence_duration))
            logging.warning(
                f"playhead leadin duration : {len(silence)} samples, {len(silence)/sr} secs"
            )
            y = np.concatenate((silence, y))


        if is_last_chunk:
            silence = np.zeros(int(sr * tail_silence_duration))
            logging.warning(
                f"playhead playout duration : {len(silence)} samples, {len(silence)/sr} secs"
            )
            y = np.concatenate((y,silence))


        logging.info("Calculate mel data")
        start_time = time.time()
        # calculate the mel data
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_ftt,
            hop_length=hop_length,
            n_mels=frame_height,
            fmin=f_low,
            fmax=f_high,
        )
        print(f"Current memory usage: {max_mem}")

        logging.info(f"Processing time: {(time.time() - start_time):.2f} seconds")

        if maxpower:
            S_dB = librosa.power_to_db(
                S, ref=maxpower, amin=np.exp(db_low / 10.0), top_db=db_high - db_low
            )
        else:
            S_dB = librosa.power_to_db(
                S, ref=np.max, amin=10 ** (db_low / 10.0), top_db=db_high - db_low
            )

        logging.info(f"Upper power : {np.max(S)}")

        # Calculate the width of the spectrogram image based on the scaling_factor

        if is_last_chunk:
            # Calculate the proportion of the last chunk relative to a full audio_buffer duration
            proportion_of_full_chunk = remaining_duration / (
                time_per_frame * mel_buffer_multiplier
            )
            # Apply this proportion to the base wide mel image width calculation
            wide_mel_image_width = int(
                (frame_width * mel_buffer_multiplier) * proportion_of_full_chunk
            )
        else:
            wide_mel_image_width = (frame_width * mel_buffer_multiplier) + frame_width

        # Plotting the spectrogram without any axes or labels
        plt.figure(figsize=(wide_mel_image_width / 100, frame_height / 100))

        # Measure processing time for mel spectrogram calculation
        # cmap=custom_cmap,
        logging.info("Render wide mel")
        start_time = time.time()
        librosa.display.specshow(
            S_dB, sr=sr, cmap=colormap, hop_length=hop_length, fmin=f_low, fmax=f_high
        )
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig("wide_mel_spectrogram.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        processing_time = time.time() - start_time
        print(f"Processing time : {processing_time:.2f} seconds")

        # slice the image
        # Load the wide mel image using PIL
        wide_mel_image = Image.open("wide_mel_spectrogram.png")

        logging.info(f"audio buffer duration: {audio_buffer} secs")
        num_frames = int(audio_buffer * frame_rate)
        logging.info(f"Number of frames in cycle : {num_frames}")

        # Get the width of the image
        image_width = (
            wide_mel_image.width - frame_width
        )  # don;t include the extended buffer
        logging.info(f"adjusted image width : {image_width}")

        step_px = image_width / num_frames  # number of pixels to slide for each frame
        logging.info(f"Pixel Step for frame : {step_px}")
        # Iterate through the wide mel image to create individual frames
        for i in range(num_frames):
            # Calculate the start and end positions for slicing
            # print(f'frame {i} of {num_frames}')
            start_pos = i * step_px
            end_pos = start_pos + frame_width

            # Slice the wide mel image to extract the frame
            frame_image = wide_mel_image.crop((start_pos, 0, end_pos, frame_height))

            playhead_overlay = create_playhead_overlay(
                total_frames_rendered + 1,
                frame_rate,
                [frame_width, frame_height],
                playhead,
                (128, 128, 128, 64),
                5,
            )

            combined_image = Image.alpha_composite(frame_image, playhead_overlay)

            # Convert the image to bytes
            cropped_frame_rgb = combined_image.convert("RGB")
            cropped_frame_bytes = cropped_frame_rgb.tobytes()

            # Write the frame bytes to ffmpeg's stdin
            ffmpeg_process.stdin.write(cropped_frame_bytes)
            total_frames_rendered += 1

            if current_position_secs == 0 and i == 0:
                # Save the frame as an image file (adjust the filename as needed)
                frame_image.save(f"frame_{i}.png")
                print("-------------------------------------SAVE")

        # increment chunk

        if current_position_secs == 0:  # first chunk
            current_position_secs += audio_buffer - (
                time_per_frame * playhead
            )  # account for offset silence
        else:
            current_position_secs += audio_buffer
        logging.critical(f"current_position = {current_position_secs}")

        if args.buffering:
            mel_buffer_multiplier = tune_buffer(
                mel_buffer_multiplier, frame_width, max_image_width_px
            )

        # reset audio buffer sizes
        audio_buffer, extended_audio_buffer = calculate_buffer(
            source_audio_duration, time_per_frame, mel_buffer_multiplier, args.buffering
        )

    # Close ffmpeg's stdin to signal end of input
    ffmpeg_process.stdin.close()

    # Wait for ffmpeg to finish encoding
    ffmpeg_process.wait()
    logging.info(f"Total frames rendered : {total_frames_rendered}")


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
        help="Audio sample rate. Overrides default (22050 Hz).",
        type=int,
        default=22050,
    )
    parser.add_argument(
        "--maxpower",
        help="Custom upper reference power for normalization, affecting spectrogram sensitivity. Overrides all auto-calculation.",
        type=float,
    )

    parser.add_argument(
        "--cpu",
        help="use CPU for processing if GPU is not available.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--buffering",
        help="Use buffering, to split the processing - for large files.",
        action="store_true",
        default=False,
    )

    parser.add_argument("-in", "--input", required=True, help="Input audio file path.")
    parser.add_argument(
        "-out",
        "--output",
        required=False,
        help='Output video file path. Default: "output.mp4".',
        default="output.mp4",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    process_audio(config, args)


if __name__ == "__main__":
    main()
