################################################################################
#                                                                              #
#                   This is the features preparation part of the project       #
#                                                                              #
################################################################################

from from_root import from_root

N_FFT = 512 #2048  # Number of FFT components for spectrograms
HOPE_LENGTH = 512  # Hop length for sliding window in spectrograms
N_MELS = 128  # Number of Mel bands for mel spectrograms
N_MFCCs = 13
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


RAW_DATASET_FOLDER = from_root('data', 'ravdess_raw')
ORGANIZED_DATASET_FOLDER = from_root('data', 'ravdess_organized')
PRODUCTION_DATASET_FOLDER = from_root('data', 'ravdess_production')
REPORT_FOLDER = from_root('reports')
import numpy as np
import pandas as pd
import librosa
from scipy import stats

# Parameters
N_MFCCs = 13  # Number of MFCCs to compute
BIN_SIZE = 100  # Size of each bin

# Define the time-domain feature names
FEATURE_NAMES = [
    "Min", "Max", "Mean", "Median", "Mode",
    "STD", "Variance", "COV", "RMS",
    "Q1", "Q2", "Q3"
]


def compute_time_domain_features(bin_matrix):
    features = []
    for i in range(bin_matrix.shape[0]):  # Iterate over each coefficient (row)
        coefficient = bin_matrix[i, :]
        min_value = np.min(coefficient)
        max_value = np.max(coefficient)
        mean_value = np.mean(coefficient)
        median_value = np.median(coefficient)
        mode_value = stats.mode(coefficient)[0]
        std_dev = np.std(coefficient)
        variance = np.var(coefficient)
        cov = std_dev / mean_value if mean_value != 0 else 0
        rms = np.sqrt(np.mean(coefficient ** 2))
        q1 = np.percentile(coefficient, 25)
        q2 = np.percentile(coefficient, 50)
        q3 = np.percentile(coefficient, 75)
        features.extend([
            min_value,
            max_value,
            mean_value,
            median_value,
            mode_value,
            std_dev,
            variance,
            cov,
            rms,
            q1,
            q2,
            q3
        ])

    return features


def bin_mfcc_matrix(mfcc_matrix):
    bins = []
    num_bins = mfcc_matrix.shape[1] // BIN_SIZE
    for i in range(num_bins):
        start = i * BIN_SIZE
        end = start + BIN_SIZE
        if end <= mfcc_matrix.shape[1]:  # Ensure we don't go out of bounds
            bins.append(mfcc_matrix[:, start:end])
    return bins


def extract_features_from_audio_file(audio_file_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCCs)[1:]  # Exclude the 0th coefficient

        # Bin the MFCC matrix
        mfcc_bins = bin_mfcc_matrix(mfcc)

        # Initialize a list to hold all features
        all_features = []

        # Extract features from each bin
        for bin_matrix in mfcc_bins:
            features = compute_time_domain_features(bin_matrix)
            all_features.append(features)

        return all_features

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None


def save_features_to_csv(features, output_csv_path):
    # Flatten the features for easier viewing
    flat_features = [item for sublist in features for item in sublist]

    # Calculate the average of each feature across all bins
    # Reshape flat_features into an array of shape (number_of_bins, number_of_features)
    num_bins = len(features)
    averaged_features = np.mean(flat_features).reshape(-1, len(FEATURE_NAMES))

    # Create a DataFrame with appropriate headers
    df = pd.DataFrame(averaged_features, columns=FEATURE_NAMES)

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Features saved to {output_csv_path}")


# Example usage
if __name__ == "__main__":
    audio_file_path = from_root("data","ravdess_production", "angry_03-01-05-01-01-01-06.wav")
    features = extract_features_from_audio_file(audio_file_path)

    if features:
        output_csv_path = 'extracted_features.csv'  # Specify your desired output CSV file path
        save_features_to_csv(features, output_csv_path)



