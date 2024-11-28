import math
import random
from pathlib import Path
from typing import Dict

import librosa  # For audio analysis and processing
import librosa.display  # For visualizing audio features
import matplotlib.pyplot as plt
import numpy as np
from from_root import from_root


# Define constants
ORGANIZED_DATASET_FOLDER = from_root('data', 'ravdess_organized')
REPORTS_DATASET_FOLDER = from_root('reports', 'figures')
TYPES_OF_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
N_FFT = 512  # Number of FFT components for spectrograms
HOPE_LENGTH = 512  # Hop length for sliding window in spectrograms
N_MELS = 128  # Number of Mel bands for mel spectrograms


def get_random_audio_files(organized_data_folder: Path, n_sample: int) -> Dict[str, list]:
    """
    Retrieve random audio files from the organized dataset grouped by emotion.
    Args:
        organized_data_folder (Path): Path to the folder containing organized audio files.
        n_sample (int): Number of samples to retrieve per emotion type.
    Returns:
        Dict[str, list]: Mapping of emotion types to a list of selected file paths.
    """
    # Dictionary to store selected audio samples by emotion type
    # audio_samples = {'calm':[file.wav,], 'angry': [file.wav] ...}
    audio_samples = {}

    # Iterate through each folder in the organized data directory
    for folder in organized_data_folder.iterdir():
        # Ensure the folder is a directory and matches a known emotion type
        if folder.is_dir() and folder.name in TYPES_OF_EMOTIONS:
            # List all files in the folder
            audio_files = list(folder.glob("*"))

            # Select up to 'n_sample' random files from the list
            selected_audio_files = random.sample(audio_files, min(n_sample, len(audio_files)))

            # Add selected audio files to the dictionary under the emotion type
            audio_samples[folder.name] = [audio_file for audio_file in selected_audio_files]

    return audio_samples


def plot_emotion_waveforms(audio_samples: Dict[str, list], num_samples: int = 6) -> None:
    """
    Plot waveforms of audio files fogrouped by emotion.
    Args:
        audio_samples (Dict[str, list]): Dict where keys are emotion types 
                                        and values are audio file paths.
        num_samples (int, optional): Number of audio files to plot per emotion. Defaults to 6.
    """
    for emotion, emotion_files in audio_samples.items():
        # Determine the grid layout of the figure
        num_rows = num_samples // 2
        num_cols = 2

        # Create subplots
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
        axs = axs.flatten()  # Flatten for easy indexing

        # Plot and save the figure of the waveforms
        for idx, emotion_file in enumerate(emotion_files[:num_samples]):
            y, _ = librosa.load(emotion_file)  # Load the audio file
            axs[idx].plot(y)
            axs[idx].set_title(f'{emotion_file.name}')
            axs[idx].set_xlabel('Time')
            axs[idx].set_ylabel('Amplitude')

        # Add a title for the current emotion
        plt.suptitle(f'Waveforms for Emotion: {emotion}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for global title
        plt.savefig(f"{REPORTS_DATASET_FOLDER}/{emotion}_wave.png")
        plt.close()


def plot_emotion_mfccs(audio_samples: Dict[str, list], num_samples: int = 6) -> None:
    """
    Plot MFCCs of audio files grouped by emotion.
    Args:
        audio_samples (Dict[str, list]): Dict where keys are emotion types 
                                        and values are audio file paths.
        num_samples (int, optional): Maximum number of audio files to plot per emotion.
                                    Defaults to 6.
    """
    for emotion, emotion_files in audio_samples.items():
        # Determine the grid layout for the current emotion
        num_rows = num_samples // 2
        num_cols = 2

        # Create subplots
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
        axs = axs.flatten()  # Flatten for easy indexing

        # Plot MFCCs
        for idx, emotion_file in enumerate(emotion_files[:num_samples]):
            y, sr = librosa.load(emotion_file)  # Load the audio file
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1:]  # Compute MFCCs

            # Plot the MFCCs
            librosa.display.specshow(mfccs, x_axis='time', ax=axs[idx], sr=sr)
            axs[idx].set_title(f'{emotion_file.name}')
            axs[idx].set_xlabel('Time')
            axs[idx].set_ylabel('MFCC Coefficients')

        # Add a title for the current emotion
        plt.suptitle(f'MFCCs for Emotion: {emotion}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for global title
        plt.savefig(f"{REPORTS_DATASET_FOLDER}/{emotion}_mfcc.png")
        plt.close()


def plot_emotion_spectrograms(audio_samples: Dict[str, list], num_samples: int = 6) -> None:
    """
    Plot spectrograms of audio files grouped by emotion.
    Args:
        audio_samples (Dict[str, list]): Dict where keys are emotion types 
                                        and values are audio file paths.
        num_samples (int, optional): Maximum number of audio files to plot per emotion.
                                    Defaults to 6.
    """
    for emotion, emotion_files in audio_samples.items():
        # Determine the grid layout for the current emotion
        num_rows = num_samples // 2
        num_cols = 2

        # Create subplots
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
        axs = axs.flatten()  # Flatten for easy indexing

        # Plot spectrograms
        for idx, emotion_file in enumerate(emotion_files[:num_samples]):
            y, sr = librosa.load(emotion_file)  # Load the audio file

            d = np.abs(librosa.stft(y=y, n_fft=N_FFT,  hop_length=HOPE_LENGTH))
            db = librosa.amplitude_to_db(d, ref=np.max)

            # Plot the spectrogram
            librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr, ax=axs[idx],
                                     fmax=8000, cmap='viridis')
            axs[idx].set_title(f'{emotion_file.name}')
            axs[idx].set_xlabel('Time')
            axs[idx].set_ylabel('Frequency')

        # Add a title for the current emotion
        plt.suptitle(f'Spectrograms for Emotion: {emotion}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for global title
        plt.savefig(f"{REPORTS_DATASET_FOLDER}/{emotion}_spectrogram.png")
        plt.close()


def plot_emotion_rms(audio_samples: Dict[str, list], num_samples: int = 6) -> None:
    """
    Plot RMS energy of audio files grouped by emotion. 

    Args:
        audio_samples (Dict[str, list]): Dict where keys are emotion types 
                                         and values are lists of audio file paths.
        num_samples (int, optional): Maximum number of audio files to plot per emotion.
                                      Defaults to 6.
    """
    for emotion, emotion_files in audio_samples.items():
        # Determine the grid layout for the current emotion
        num_rows = num_samples // 2
        num_cols = 2

        # Create subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
        axs = axs.flatten()  # Flatten for easy indexing

        # Plot RMS energy
        for idx, emotion_file in enumerate(emotion_files[:num_samples]):
            # Load the audio file
            y, sr = librosa.load(emotion_file)

            # Compute RMS energy
            rms = librosa.feature.rms(y=y)[0]  # Extract the first array from the result
            times = librosa.times_like(rms, sr=sr)  # Create time axis for RMS values

            # Plot the RMS energy
            axs[idx].plot(times, rms, label="RMS Energy", color='purple')
            axs[idx].set_title(f'RMS of {emotion_file.name}')
            axs[idx].set_xlabel('Time (s)')
            axs[idx].set_ylabel('RMS Energy')
            axs[idx].legend()

        # Adjust layout and global title
        fig.suptitle(f'RMS Energy for Emotion: {emotion}', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the global title
        plt.savefig(f"{REPORTS_DATASET_FOLDER}/{emotion}_rms.png")
        plt.close()


def plot_emotion_mel_spectrogram(audio_samples: Dict[str, list], num_samples: int = 6) -> None:
    """
    Plot mel spectrograms of audio files grouped by emotion.
    Args:
        audio_samples (Dict[str, list]): Dict where keys are emotion types 
                                         and values are lists of audio file paths.
        num_samples (int, optional): Maximum number of audio files to plot per emotion.
                                      Defaults to 6.
    """
    for emotion, emotion_files in audio_samples.items():
        # Determine the grid layout for the current emotion
        num_rows = num_samples // 2
        num_cols = 2

        # Create subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
        axs = axs.flatten()  # Flatten for easy indexing

        # Plot log power mel spectrograms
        for idx, emotion_file in enumerate(emotion_files[:num_samples]):
            # Load the audio file
            y, sr = librosa.load(emotion_file)

            # Compute the mel spectrogram
            s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                               hop_length=HOPE_LENGTH, n_mels=128)

            # Convert the mel spectrogram to log scale (decibels)
            s_db = librosa.power_to_db(S=s, ref=np.max)

            # Plot the log power mel spectrogram
            img = librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr=sr,
                                           fmax=8000, ax=axs[idx], cmap='viridis')
            axs[idx].set_title(f'{emotion_file.name}')
            axs[idx].set_xlabel('Time')
            axs[idx].set_ylabel('Mel Frequency (Hz)')
            fig.colorbar(img, ax=axs[idx], format='%+2.0f dB')

        # Add a title for the current emotion
        fig.suptitle(f'Mel Spectrograms for Emotion: {emotion}', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the global title
        plt.savefig(f"{REPORTS_DATASET_FOLDER}/{emotion}_mel_spectrogram.png")
        plt.close()


def plot_audio_features(audio_samples: Dict[str, list], n_samples: int = 6) -> None:
    """
    Calls all plotting functions for various audio features.
    Args:
        audio_samples (dict): A dictionary where keys are emotion labels 
                              and values are lists of audio file paths.
        n_samples (int): Maximum number of audio files to plot per emotion.
                        Defaults to 6.
    """
    # Call the function that plots the waveforms
    plot_emotion_waveforms(audio_samples=audio_samples, num_samples=n_samples)

    # Call the function that plots the MFCCs
    plot_emotion_mfccs(audio_samples=audio_samples, num_samples=n_samples)

    # Call the function that plots the RMS
    plot_emotion_rms(audio_samples=audio_samples, num_samples=n_samples)

    # Call the function that plots the spectrograms
    plot_emotion_spectrograms(audio_samples=audio_samples, num_samples=n_samples)

    # Call the function that plots the log mel spectrograms
    plot_emotion_mel_spectrogram(audio_samples=audio_samples, num_samples=n_samples)
