import subprocess
import time
import os
import logging
import argparse
import pprint

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


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


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def process_audio(config, args):
    # extract the config and argumates to variables.
    audio_fsp = args.input
    video_fsp = args.output
    logging.info(f"Audio file in : {audio_fsp}")
    logging.info(f"Video file out: {video_fsp}")


    sr = args.sr    
    audio_start = args.start
    audio_duration = args.duration
    #TODO    
    """ logging.info(f"Audio start offset (secs) : {audio_start}")
        if audio_duration:
            logging.info(f"Audio clip duration (secs) : {audio_duration}") """

    if args.maxpower:
        maxpower = args.maxpower
    else:
        maxpower = config.get("mel_spectrogram", {}).get("maxpower", None)

    ##loads the file
    total_duration = librosa.get_duration(path=audio_fsp)
    logging.info(f"Total duration (secs): {total_duration}")

    logging.info(f"Resample rate : {sr} hz")
    logging.info(f"Upper Reference Power : {maxpower}")

    # get the output video paramters from config
    frame_width = config.get("video", {}).get("width", 800)
    frame_height = config.get("video", {}).get("height", 200)
    frame_rate = config.get("video", {}).get("frame_rate", 30)
    logging.info(f"Output Video : {frame_width}px x {frame_height}px @ {frame_rate}fps ")

    # get the cofig used to create the mel scale spectrograms
    #   the color pallate
    #   https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colormap = config.get("audio_visualization", {}).get("cmap", "magma")
    if colormap not in plt.colormaps():
        logging.warning("'Colormap not found, using 'magma'")
        colormap = "magma"

    # get the playhead position from config.
    # 0 is hard left of the frame, 0.5 is centre
    playhead = config.get("audio_visualization", {}).get("playhead_position", 0.0)


    logging.info(f"Spectrogram Pallette : {colormap} ")

    # config for the spectrogram
    #   time_per_frame, is the duration of aution represented in 1 x frame_width
    time_per_frame = config["audio_visualization"]["time_per_frame"]

    #   a multiplyer  that determines how much audio to process each cycle
    #   smaller numbers will take longer, higher numbers will use a lot of memory
    #   1 minute displayed in a frame x 20 = 20 minutes of audio processed
    mel_buffer_multiplier = config["audio_visualization"]["mel_buffer_multiplier"]

    audio_buffer = (
        time_per_frame * mel_buffer_multiplier
    )  # the ammount of audio data needed to produce the wide mel buffer
    extended_audio_buffer = (
        audio_buffer + time_per_frame
    )  # we add enough to service the sliding window tail

    logging.info(f"Full frame duration : {time_per_frame} secs")
    logging.info(f"Audio Buffer : {audio_buffer} secs")

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

    ### Start the ffmpeg process
    ffmpeg_cmd = [
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
        "18",  # Constant rate factor for quality
        "-preset",
        "fast",  # Preset for encoding speed
        "-vf",
        "format=yuv420p",  # Pixel format for compatibility
        f"{video_fsp}",
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    ##### start work
    current_position_secs = 0
    total_frames_rendered = 0
    while current_position_secs < total_duration:
        logging.info(f"Starting Chunk")
        logging.critical(f'current_position = {current_position_secs}')
        ## check if this is the last chunk
        is_last_chunk = (current_position_secs + audio_buffer) >= total_duration
        if is_last_chunk:
            # resize audio buffer to match remaining audio
            remaining_duration = total_duration - current_position_secs
            audio_buffer = remaining_duration
            extended_audio_buffer = remaining_duration

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
            silence = np.zeros(int(sr * time_per_frame * playhead))
            logging.warning(f'playhead offset : {len(silence)} samples, {len(silence)/sr} secs')
            y = np.concatenate((silence, y))

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
        logging.info(f"Processing time: {(time.time() - start_time):.2f} seconds")

        if maxpower:
            S_dB = librosa.power_to_db(
                S, ref=maxpower, amin=np.exp(db_low / 10.0), top_db=db_high - db_low
            )
        else:
            S_dB = librosa.power_to_db(
                S, ref=np.max, amin=10 ** (db_low / 10.0), top_db=db_high - db_low
            )

        logging.info(f'Upper power : {np.max(S)}')

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

        logging.info(f'audio buffer duration: {audio_buffer} secs')
        num_frames = int(audio_buffer * frame_rate)
        logging.info(f'Number of frames in cycle : {num_frames}')

        # Get the width of the image
        image_width = (
            wide_mel_image.width - frame_width
        )  # don;t include the extended buffer
        logging.info(f'adjusted image width : {image_width}')

        step_px = image_width / num_frames  # number of pixels to slide for each frame
        logging.info(f'Pixel Step for frame : {step_px}')
        # Iterate through the wide mel image to create individual frames
        for i in range(num_frames):
            # Calculate the start and end positions for slicing
            # print(f'frame {i} of {num_frames}')
            start_pos = i * step_px
            end_pos = start_pos + frame_width

            # Slice the wide mel image to extract the frame
            frame_image = wide_mel_image.crop((start_pos, 0, end_pos, frame_height))

            # Convert the image to bytes
            cropped_frame_rgb = frame_image.convert("RGB")
            cropped_frame_bytes = cropped_frame_rgb.tobytes()

            # Write the frame bytes to ffmpeg's stdin
            ffmpeg_process.stdin.write(cropped_frame_bytes)
            total_frames_rendered+=1

            if current_position_secs == 0 and i==0:
            # Save the frame as an image file (adjust the filename as needed)
                frame_image.save(f'frame_{i}.png')
                print("-------------------------------------SAVE")

        # increment chunk

        if current_position_secs == 0:  # first chunk
            current_position_secs += audio_buffer - (time_per_frame * playhead)  # account for offset silence
        else:
            current_position_secs += audio_buffer
        logging.critical(f'current_position = {current_position_secs}')

    # Close ffmpeg's stdin to signal end of input
    ffmpeg_process.stdin.close()

    # Wait for ffmpeg to finish encoding
    ffmpeg_process.wait()
    logging.info(f'Total frames rendered : {total_frames_rendered}')

def main():
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

    # Pretty print using pformat to get a string representation
    formatted_config = pprint.pformat(config, indent=4)
    logging.info(f"Config Loaded: {formatted_config}")

    process_audio(config, args)


if __name__ == "__main__":
    main()
