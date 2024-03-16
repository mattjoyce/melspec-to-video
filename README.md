
Overview

This tool generates a scrolling spectrogram video from an audio file. It's designed to visually represent the spectrum of frequencies in the audio as they vary over time, with the spectrogram scrolling horizontally to match the playback. This visualization aids in the analysis and presentation of audio characteristics in a dynamic and engaging format.
Goals

* Visual Analysis: Provide a detailed visual representation of the audio's frequency spectrum over time.
* Customization: Allow users to configure various parameters such as frequency range, the size of the video, and visualization palette.
* Efficiency: Streamline the process to handle long audio files efficiently, balancing detail with processing requirements.

Features

* Configurable spectrogram resolution and frequency range.
* Adjustable dynamic range (dB) for amplitude visualization.
* Mel scale frequency representation for perceptually-relevant analysis.
* Configurable palette for frequency amplitude mapping.
* Control over the scrolling speed of the spectrogram, allowing users to specify the duration for the spectrogram to scroll across the screen.
* CLI interface for easy operation and integration into workflows.

Usage
Prerequisites

* Python 3.8 or newer
* Required Python packages: numpy, librosa, matplotlib, PyYAML
* ffmpeg installed and available in the system's PATH for video encoding.

Installation

*Clone the repository or download the source code.
* Install the required Python packages:
* pip install numpy librosa matplotlib PyYAML

Configuration

Edit the config.yaml file to specify the spectrogram generation parameters:

Running the Tool

Execute the script from the command line, specifying the configuration file, input audio file, and output video file:

python RenderScrollingSpectrogram.py -c config.yaml -in input_audio.wav -out output_video.mp4

![Example GIF](example.gif)

Contributing

Contributions to improve the tool or extend its capabilities are welcome. Please follow the standard GitHub pull request process for contributions.
License

Specify the license under which the software is released.