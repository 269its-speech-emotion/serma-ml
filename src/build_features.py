################################################################################
#                                                                              #
#                   This is the features preparation part of the project       #
#                                                                              #
################################################################################

from pathlib import Path
from typing import Dict

import librosa  # For audio analysis and processing
import numpy as np
from tqdm import tqdm


N_FFT = 512  # Number of FFT components for spectrograms
HOPE_LENGTH = 512  # Hop length for sliding window in spectrograms
N_MELS = 128  # Number of Mel bands for mel spectrograms


def pad_to_consistent_shape(arr, target_shape) -> np.ndarray :
    """
    Pads a NumPy array 'arr' to have the specified 'target_shape'
    Args:
        arr (_type_): actual feature array
        target_shape (_type_): Targeted shape
    Returns:
        np.ndarray: added NumPy array
    """

    current_shape = arr.shape
    pad_height = target_shape[0] - current_shape[0]
    pad_width = target_shape[1] - current_shape[1]

    # Ensure that padding values are non-negative
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)

    padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width)), mode='constant')

    return padded_arr


def normalize_feature(feature):
    """
    Normalize a single feature array between 0 and 1.
    Args:
        feature (np.ndarray): Feature array to normalize.
    Returns:
        np.ndarray: Normalized feature array.
    """
    min_val, max_val = np.min(feature), np.max(feature)
    normalized_feature = (feature - min_val) / (max_val - min_val)
    return normalized_feature


def pad_features(features: Dict[str, list], max_time_steps: Dict[str, int]) -> Dict[str, list]:
    """
    Pad audio features to ensure consistent shapes across all samples.
    Args:
        features (Dict[str, list]): Dict of feature types and their corresponding lists of arrays.
        max_time_steps (Dict[str, int]): Maximum time steps for each feature type.
    Returns:
        Dict[str, list]: Dictionary with padded features for each feature type.
    """
    padded_features = {}
    for feature_type, feature_list in features.items():
        if feature_type == 'rms':
            # Pad 1D features(RMS energy)
            padded_features[feature_type] = [
                np.pad(f, (0, max_time_steps[feature_type] - len(f)), mode='constant') for f in feature_list
            ]
        else:
            # Pad 2D features (e.g., spectrograms, MFCCs)
            target_shape = (feature_list[0].shape[0], max_time_steps[feature_type])
            padded_features[feature_type] = [
                pad_to_consistent_shape(f, target_shape) for f in feature_list
            ]
    return padded_features


def extract_audio_features(organized_data_folder: Path) -> Dict:
    """
    Extracts features (mel log spectrogram, spectrogram, MFCC, RMS) and organizes them in a dict.
    Args:
        organized_data_folder (Path): Path to the target directory for the organized dataset.
    Returns:
        Dict: Dictionary with labels and extracted features for each feature type.
    """
    labels = []  # Centralized storage for emotion labels
    features = {  # Initialize feature containers for each type
        'mel_log_spectrogram': [],
        'spectrogram': [],
        'mfcc': [],
        'rms': []
    }

    # Iterate through each emotion folder
    for folder in organized_data_folder.iterdir():
        if folder.is_dir():  # Ensure it's a directory
            for emotion_file in tqdm(folder.iterdir(), desc=f'Extract Features for {folder.name}'):
                try:
                    # Load audio
                    y, sr = librosa.load(emotion_file)

                    # Extract mel spectrogram and convert to log scale
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                                              hop_length=HOPE_LENGTH,
                                                              n_mels=N_MELS)
                    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    # Normalize log mel spectrogram
                    normalized_log_mel_spec = normalize_feature(feature=log_mel_spec)

                    # Extract spectrogram (log scale)
                    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOPE_LENGTH))
                    log_spec = librosa.amplitude_to_db(spec, ref=np.max)
                    # Normalize spectrogram
                    normalized_log_spec = normalize_feature(feature=log_spec)

                    # Extract MFCCs (Exclude the 0th coefficient)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1:]
                    # Normalize mfcc
                    normalized_mfcc = normalize_feature(feature=mfcc)

                    # Compute RMS energy
                    rms = librosa.feature.rms(y=y)[0]
                    # Normalize rms
                    normalized_rms = normalize_feature(feature=rms)

                    # Store features
                    features['mel_log_spectrogram'].append(normalized_log_mel_spec)
                    features['spectrogram'].append(normalized_log_spec)
                    features['mfcc'].append(normalized_mfcc)
                    features['rms'].append(normalized_rms)
                    labels.append(folder.name)

                except Exception as e:
                    print(f"Error processing {emotion_file}: {e}")

    # Padding features to ensure consistency in shape
    print("Padding features to consistent shapes...")
    max_time_steps = {
        'mel_log_spectrogram': max(f.shape[1] for f in features['mel_log_spectrogram']),
        'spectrogram': max(f.shape[1] for f in features['spectrogram']),
        'mfcc': max(f.shape[1] for f in features['mfcc']),
        'rms': max(len(f) for f in features['rms'])
    }
    padded_features = pad_features(features, max_time_steps)

    # Prepare the final dataset dictionary
    total_dataset = {
        'labels': labels,
        'data': padded_features
    }

    return total_dataset
