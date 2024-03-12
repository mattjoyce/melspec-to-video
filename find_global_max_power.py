import argparse
import librosa
import yaml
import numpy as np
import time
import logging
import sys
import soundfile as sf

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_resample(audio_path, target_sr=22050):
    # Load and resample audio file
    data, samplerate = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Convert to mono
    data_resampled = librosa.resample(data, orig_sr=samplerate, target_sr=target_sr)
    return data_resampled, target_sr

def load_config(config_path):
    # Load YAML configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    return config

def process_audio_chunk(y, sr, config):
    # Process audio chunk and return maximum power
    S = librosa.feature.melspectrogram(y=y, sr=sr, 
                                       n_fft=config['mel_spectrogram']['n_fft'], 
                                       hop_length=config['mel_spectrogram']['hop_length'],
                                       n_mels=config['video']['height'],
                                       fmin=config['mel_spectrogram']['f_low'],
                                       fmax=config['mel_spectrogram']['f_high'])
    return np.max(S)

def process_audio(config_path, audio_path, start, duration, target_sr, chunk_duration):
    config = load_config(config_path)
    logging.info("Starting audio processing")

    y, sr = load_and_resample(audio_path, target_sr)

    # Handle chunk processing
    total_samples = len(y)
    samples_per_chunk = int(sr * chunk_duration)
    max_power_list = []

    for i in range(0, total_samples, samples_per_chunk):
        y_chunk = y[i:i+samples_per_chunk]
        chunk_max_power = process_audio_chunk(y_chunk, sr, config)
        max_power_list.append(chunk_max_power)
        logging.info(f"Chunk max power: {chunk_max_power}")

    overall_max_power = max(max_power_list)
    logging.info(f"Overall max power: {overall_max_power}")
    # Update the configuration accordingly
    config = update_config(config, config_path, audio_path, overall_max_power)

def update_config(config, config_path, audio_path, global_max_power):
    # Update YAML configuration with global max power
    audio_files_config = config.get('audio_files', {})
    audio_files_config[audio_path] = {'global_max_power': int(global_max_power)}
    config['audio_files'] = audio_files_config
    try:
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file)
        logging.info("Configuration updated successfully.")
    except Exception as e:
        logging.error(f"Failed to update configuration: {e}")
        sys.exit(1)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Processing Tool")
    parser.add_argument("--config", help="Path to the configuration YAML file", required=True)
    parser.add_argument("--audio", help="Path to the audio file to process", required=True)
    parser.add_argument("--start", help="Start time in seconds", type=float, default=0)
    parser.add_argument("--duration", help="Duration in seconds", type=float)
    parser.add_argument("--sr", help="Sample rate to override librosa's default", type=int, default=22050)
    parser.add_argument("--chunk_duration", help="Duration of each chunk in seconds", type=float, default=None)

    args = parser.parse_args()

    # Check for chunk processing
    if args.chunk_duration:
        process_audio(args.config, args.audio, args.start, args.duration, args.sr, args.chunk_duration)
    else:
        process_audio(args.config, args.audio, args.start, args.duration, args.sr, float('inf'))  # Treat as a single chunk if no chunk_duration provided
