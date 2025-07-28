from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from config import logger
from src.extract_features import extract_mfcc_features, extract_mfcct_features
from src.preprocessing import preprocess_audio_file


def pad_to_consistent_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Pads a NumPy array 'arr' to have the specified 'target_shape'.

    Args:
        arr (np.ndarray): Actual feature array.
        target_shape (tuple): Targeted shape.

    Returns:
        np.ndarray: Padded NumPy array.
    """
    current_shape = arr.shape
    pad_height = target_shape[0] - current_shape[1]  # Adjust for time steps
    pad_width = target_shape[1] - current_shape[2]  # Adjust for coefficients

    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)

    # Note: Padding along time steps (axis 1) and coefficients (axis 2)
    padded_arr = np.pad(arr, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')

    return padded_arr


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
        # Determine the target shape based on feature type
        if feature_type == 'mfcc':
            target_shape = (1, max_time_steps[feature_type], feature_list[0].shape[2])  # (batch, time, coeff)
        elif feature_type == 'mfcct':
            target_shape = (1, max_time_steps[feature_type], feature_list[0].shape[2])  # Ensure batch and coeff match

        padded_features[feature_type] = [
            pad_to_consistent_shape(f, target_shape) for f in feature_list
        ]
    return padded_features


def build_features(organized_data_folder: Path) -> Dict:
    """
    Builds a dataset of features from organized audio data.

    Args:
        organized_data_folder (Path): Path to the folder containing organized audio data.

    Returns:
        Dict: Dictionary containing labels and padded feature data.
    """
    features = {
        'labels': [],  # Emotion labels
        'data': {
            'mfcc': [],  # MFCC features
            'mfcct': []  # MFCCT features
        }
    }

    # Iterate through each emotion folder
    for folder in organized_data_folder.iterdir():
        if folder.is_dir():  # Ensure it's a directory
            for audio_file in tqdm(folder.iterdir(), desc=f'Extract Features for {folder.name}'):
                try:
                    # 1. Preprocess the audio file
                    preprocessed_signal = preprocess_audio_file(audio_file)

                    # 2. Extract MFCC features
                    mfcc = extract_mfcc_features(preprocessed_signal, frame_length=25, frame_step=10)

                    # 3. Extract MFCCT features
                    mfcct = extract_mfcct_features(mfcc, bin_size=1500)

                    logger.info(f"MFCC shape: {mfcc.shape}, MFCCT shape: {mfcct.shape}")

                    features['data']['mfcc'].append(mfcc)
                    features['data']['mfcct'].append(mfcct)
                    features['labels'].append(folder.name)
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")

    # Log info after feature extraction is complete
    logger.info(f"All features extracted from {len(features['labels'])} audio files.")
    logger.info(f"MFCC: {len(features['data']['mfcc'])}")
    logger.info(f"MFCCT: {len(features['data']['mfcct'])}")

    # Padding features to ensure consistency in shape
    logger.info("Padding features to consistent shapes...")
    max_time_steps = {
        'mfcc': max((f.shape[1] for f in features['data']['mfcc']), default=0),
        'mfcct': max((f.shape[1] for f in features['data']['mfcct']), default=0),
    }

    logger.info(f"max_time_steps , {max_time_steps}")

    padded_features = pad_features(features['data'], max_time_steps)

    # Prepare the final dataset dictionary
    total_dataset = {
        'labels': features['labels'],
        'data': padded_features
    }

    return total_dataset

def normalize_feature(feature: np.ndarray) -> np.ndarray:
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


def prepare_features_for_cnn(features: Dict[str, list]) -> tuple:
    """
    Prepares MFCC and MFCCT features for CNN input from already padded features.

    Args:
        features (Dict[str, list]): Dictionary containing 'data' with MFCC and MFCCT feature arrays.

    Returns:
        tuple: Stacked and normalized MFCC and MFCCT features ready for CNN input.
    """
    # Extract padded MFCC and MFCCT features
    mfcc_data = np.array(features['data']['mfcc'])
    mfcct_data = np.array(features['data']['mfcct'])

    # Normalize MFCC features using the normalize_feature function
    mfcc_data = np.array([normalize_feature(mfcc) for mfcc in mfcc_data])

    # Normalize MFCCT features using the normalize_feature function
    mfcct_data = np.array([normalize_feature(mfcct) for mfcct in mfcct_data])

    return mfcc_data, mfcct_data
