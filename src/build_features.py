from pathlib import Path
from typing import Dict

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # J48 (C4.5-like)
from tqdm import tqdm

from src.config import logger
from src.extract_features import extract_mfcc_features, extract_mfcct_features


def pad_to_consistent_shape(arr, target_shape):
    """
    Pads a NumPy array 'arr' to have the specified 'target_shape' only in the time dimension.
    Args:
        arr (np.ndarray): Actual feature array with shape (n_mfcc, time_steps)
        target_shape (tuple): Target shape (n_mfcc, max_time_steps)
    Returns:
        np.ndarray: Padded NumPy array
    """
    current_shape = arr.shape
    pad_height = target_shape[0] - current_shape[0]  # Number of MFCC coefficients (should be 0)
    pad_width = target_shape[1] - current_shape[1]  # Number of time steps to pad

    # Ensure that padding is only applied to the second dimension (time steps)
    pad_height = max(0, pad_height)  # Should ideally be 0 if MFCC dimension is fixed
    pad_width = max(0, pad_width)

    padded_arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')  # Only pad time axis

    return padded_arr




def pad_features(features_dict, max_time_steps):
    """Pads MFCC and MFCCT features to ensure uniform shapes."""
    padded_data = {'mfcc': [], 'mfcct': []}

    for feature_name in ['mfcc', 'mfcct']:
        max_length = max_time_steps[feature_name]
        logger.info(f"Padding {feature_name} to max length: {max_length}")

        for feature in features_dict[feature_name]:
            feature_shape = feature.shape

            # Ensure feature has correct dimensions before padding
            if len(feature.shape) == 1:
                feature = np.expand_dims(feature, axis=0)

            # Correct time axis padding
            pad_width = [(0, 0), (0, max_length - feature_shape[1])]
            padded_feature = np.pad(feature, pad_width, mode='constant')

            padded_data[feature_name].append(padded_feature)

    return {key: np.array(value) for key, value in padded_data.items()}



def build_features(preprocessed_data_folder: Path, sample_rate=16000) -> Dict:
    """
    Builds a dataset of features from preprocessed audio data.

    Args:
        preprocessed_data_folder (Path): Path to the folder containing preprocessed audio files.
        sample_rate (int): Target sample rate for loading audio.

    Returns:
        Dict: Dictionary containing labels and feature data.
    """
    features_data = {
        'labels': [],
        'data': {
            'mfcc': [],
            'mfcct': []
        }
    }

    # Iterate through each emotion folder
    for folder in preprocessed_data_folder.iterdir():
        if folder.is_dir():  # Ensure it's a directory
            for audio_file in tqdm(folder.iterdir(), desc=f'Extracting Features for {folder.name}'):
                try:
                    # Load preprocessed audio (should already be cleaned and saved as WAV)
                    preprocessed_signal, sr = librosa.load(audio_file, sr=sample_rate, mono=True)
                    logger.info(f"Loaded preprocessed signal: length={len(preprocessed_signal)}, rate={sr} Hz")

                    # Ensure the signal isn't empty
                    if len(preprocessed_signal) == 0:
                        logger.warning(f"Skipping empty file: {audio_file}")
                        continue

                    # Extract MFCC features
                    mfcc = extract_mfcc_features(preprocessed_signal, frame_length=25, frame_step=10)

                    # Extract MFCCT features
                    mfcct = extract_mfcct_features(mfcc, bin_size=1500)

                    logger.info(f"MFCC shape: {mfcc.shape}, MFCCT shape: {mfcct.shape}")

                    # Append features and labels
                    features_data['data']['mfcc'].append(mfcc)
                    features_data['data']['mfcct'].append(mfcct)
                    features_data['labels'].append(folder.name)

                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")

    return features_data


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

    print("MFCC features data shape:", mfcc_data.shape)
    print("MFCCT features data shape:", mfcc_data.shape)

    time_steps, num_features = mfcc_data.shape[2], mfcc_data.shape[3]
    mfcc_data_reshaped = mfcc_data.reshape(mfcc_data.shape[0], time_steps, num_features)

    time_steps, num_features = mfcct_data.shape[2], mfcct_data.shape[3]
    mfcct_data_reshaped = mfcct_data.reshape(mfcct_data.shape[0], time_steps, num_features)

    return mfcc_data_reshaped, mfcct_data_reshaped


def prepare_mfcct_for_ml(mfcct_data, normalize=True):
    """Prepare MFCCT features for traditional ML classifiers by flattening and normalizing.

    Args:
        mfcct_data (np.ndarray): MFCCT features (shape: n_samples, timesteps, features, e.g., (195, 48, 13)).
        normalize (bool): Whether to normalize features to [0, 1] (default: True).

    Returns:
        tuple: Flattened features (n_samples, n_features) and original shape info.
    """
    # Ensure input is 3D
    if len(mfcct_data.shape) != 3:
        raise ValueError(f"Expected 3D MFCCT data, got shape {mfcct_data.shape}")

    n_samples, timesteps, features = mfcct_data.shape
    logger.info(f"Preparing MFCCT data: shape={mfcct_data.shape}, samples={n_samples}, timesteps={timesteps}, features={features}")

    # Flatten to 2D: (n_samples, timesteps * features)
    flattened_data = mfcct_data.reshape(n_samples, -1)  # Shape: (n_samples, timesteps * features)
    logger.info(f"Flattened MFCCT shape: {flattened_data.shape}")

    # Normalize if requested (global normalization across dataset)
    if normalize:
        min_val, max_val = np.min(flattened_data), np.max(flattened_data)
        if max_val > min_val:
            flattened_data = (flattened_data - min_val) / (max_val - min_val)
            logger.info("Normalized MFCCT features to [0, 1]")
        else:
            logger.warning("No normalization applied: min and max values are equal")

    return flattened_data, (timesteps, features)  # Return original shape for potential reconstruction


def build_ml_classifiers(mfcct_data, labels, test_size=0.2, random_state=42):
    """Build and evaluate ML classifiers (k-NN, RF, J48, NB, SVM) on MFCCT features.

    Args:
        mfcct_data (np.ndarray): MFCCT features (shape: n_samples, timesteps, features).
        labels (np.ndarray): Numerical labels for emotions (e.g., 0–7).
        test_size (float): Proportion of dataset for testing (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        dict: Classifier models and their performance metrics.
    """
    # Prepare features for ML
    X_flattened, _ = prepare_mfcct_for_ml(mfcct_data)
    logger.info(f"Prepared ML features shape: {X_flattened.shape}")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, labels, test_size=test_size,
                                                        random_state=random_state, stratify=labels)
    logger.info(f"Train/test split: train={len(y_train)}, test={len(y_test)}")

    # Define classifiers
    classifiers = {
        'k-NN': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'J48': DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=random_state),  # C4.5-like
        'NB': GaussianNB(),
        'SVM': SVC(kernel='rbf', random_state=random_state)
    }

    results = {}
    for name, clf in classifiers.items():
        try:
            # Train the classifier
            clf.fit(X_train, y_train)
            logger.info(f"Trained {name} classifier")

            # Predict on test set
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Cross-validation (5-fold)
            cv_scores = cross_val_score(clf, X_flattened, labels, cv=5, scoring='accuracy')
            cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

            # Store results
            results[name] = {
                'model': clf,
                'accuracy': accuracy,
                'classification_report': report,
                'cv_mean_accuracy': cv_mean,
                'cv_std': cv_std
            }
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f}, CV Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")

        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")

    return results