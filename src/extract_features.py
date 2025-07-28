################################################################################
#                                                                              #
#                    Feature Extraction Component for SER Project              #
#                   Extracts MFCC and MFCCT features from preprocessed audio   #
#                   for emotion recognition using CNN-based methods.           #
#                                                                              #
################################################################################

import librosa
import numpy as np
from scipy.fft import dct
from scipy.stats import mode

from config import logger


def extract_mfcc_features(audio_signal, sample_rate=16000, frame_length=25, frame_step=10, n_mfcc=13, n_mels=26):
    """Extract MFCC features from audio, per the paper's Section 3.3(a).

    Args:
        audio_signal (np.ndarray): Preprocessed audio (shape: n_samples).
        sample_rate (int): Audio sample rate (default: 16000 Hz).
        frame_length (int): Frame length in ms (default: 25 ms).
        frame_step (int): Frame step in ms (default: 10 ms).
        n_mfcc (int): Number of MFCC coefficients (default: 13).
        n_mels (int): Number of Mel filter banks (default: 26).

    Returns:
        np.ndarray: MFCC features (shape: 1, n_frames, n_mfcc).
    """
    # Convert frame params to samples
    hop_length = int(sample_rate * frame_step / 1000)  # Step: 10 ms = 160 samples
    win_length = int(sample_rate * frame_length / 1000)  # Length: 25 ms = 400 samples

    # Check signal length for framing
    if len(audio_signal) < win_length:
        raise ValueError(f"Signal too short: {len(audio_signal)} < {win_length}")

    # Frame audio with 25 ms frames, 10 ms overlap
    frames = librosa.util.frame(audio_signal, frame_length=win_length, hop_length=hop_length)
    logger.info(f"Framed audio shape: {frames.shape}")

    # Apply Hamming window to smooth edges
    hamming_window = np.hamming(win_length)
    windowed_frames = frames * hamming_window[:, np.newaxis]
    logger.info(f"Windowed frames shape: {windowed_frames.shape}")

    # Compute STFT and magnitude spectrum
    n_fft = win_length  # Match frame length
    stft = librosa.stft(windowed_frames, n_fft=n_fft, hop_length=hop_length)
    logger.info(f"STFT shape: {stft.shape}")
    magnitude_spectrum = np.abs(stft)

    # Log magnitude for Mel scaling
    log_magnitude_spectrum = np.log(magnitude_spectrum + 1e-8)  # Avoid log(0)
    logger.info(f"Log magnitude shape: {log_magnitude_spectrum.shape}")

    # Apply Mel filter banks and log
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_spectrum = np.dot(mel_basis, magnitude_spectrum)
    logger.info(f"Mel spectrum shape: {mel_spectrum.shape}")
    log_mel_spectrum = np.log(mel_spectrum + 1e-8)

    # Compute MFCCs via DCT
    mfcc = dct(log_mel_spectrum, axis=0, type=2, norm='ortho')[:n_mfcc, :]
    logger.info(f"MFCC shape before transpose: {mfcc.shape}")

    # Transpose and ensure batched output
    mfcc_features = mfcc.T  # Frames as rows, coefficients as columns
    logger.info(f"MFCC shape after transpose: {mfcc_features.shape}")
    mfcc_features = np.atleast_3d(mfcc_features)  # Ensure (1, n_frames, n_mfcc)

    if mfcc_features.shape[0] != 1:
        raise ValueError(f"Unexpected batch size: {mfcc_features.shape[0]}")

    logger.info(f"Final MFCC shape: {mfcc_features.shape}")
    return mfcc_features


def compute_mode_or_fallback(bin_data):
    """Compute mode of MFCC bin data, falling back to mean if mode fails.

    Args:
        bin_data (np.ndarray): MFCC segment (shape: (bin_size, 1)).

    Returns:
        float: Mode or mean value.
    """
    try:
        mode_value = mode(bin_data, axis=0, keepdims=False).mode[0].item()
    except Exception as e:
        logger.warning(f"Mode error for shape {bin_data.shape}: {str(e)}. Using mean: {np.mean(bin_data, axis=0).item()}")
        mode_value = np.mean(bin_data, axis=0).item()
    return mode_value


def compute_time_domain_features(bin_data):
    """Compute 12 time-domain features for an MFCC bin.

    Args:
        bin_data (np.ndarray): MFCC segment (shape: (bin_size, 1)).

    Returns:
        np.ndarray: 12 features: MIN, MAX, Mean, Median, Mode, STD, VAR, COV, RMS, Q1, Q2, Q3.
    """
    # Init array for features
    features = np.zeros(12)

    # Compute features
    features[0] = np.min(bin_data, axis=0).item()  # MIN
    features[1] = np.max(bin_data, axis=0).item()  # MAX
    features[2] = np.mean(bin_data, axis=0).item()  # Mean
    features[3] = np.median(bin_data, axis=0).item()  # Median
    features[4] = compute_mode_or_fallback(bin_data)  # Mode
    features[5] = np.std(bin_data, axis=0).item()  # STD
    features[6] = np.var(bin_data, axis=0).item()  # VAR
    features[7] = np.var(bin_data, axis=0).item()  # COV
    features[8] = np.sqrt(np.mean(bin_data ** 2, axis=0)).item()  # RMS
    features[9] = np.percentile(bin_data, 25, axis=0).item()  # Q1
    features[10] = np.percentile(bin_data, 50, axis=0).item()  # Q2
    features[11] = np.percentile(bin_data, 75, axis=0).item()  # Q3

    # Log feature summary
    logger.debug(f"Computed features for shape {bin_data.shape}: {features}")

    return features


def extract_mfcct_features(mfcc_features, bin_size=1500, time_domain_features=12):
    """Extract MFCCT features by binning MFCCs and computing time-domain stats.

    Args:
        mfcc_features (np.ndarray): MFCC data (shape: batch_size, n_frames, n_mfcc).
        bin_size (int): Rows per bin (default: 1500).
        time_domain_features (int): Number of features per bin (default: 12).

    Returns:
        np.ndarray: MFCCT features (shape: batch_size, n_bins * 12, n_mfcc).
    """
    # Validate input
    if len(mfcc_features.shape) != 3:
        raise ValueError(f"Expected 3D MFCC features, got {mfcc_features.shape}")

    batch_size, n_frames, n_mfcc = mfcc_features.shape
    logger.info(f"Processing batch: size={batch_size}, frames={n_frames}, mfcc={n_mfcc}")

    mfcct_batch = []

    for batch_idx in range(batch_size):
        # Get MFCC data for batch
        current_mfcc = mfcc_features[batch_idx]
        n_bins = n_frames // bin_size

        if n_frames < bin_size:
            bin_size = n_frames
            n_bins = 1
            logger.info(f"Adjusted bin_size to {bin_size} for batch {batch_idx}")

        if n_frames % bin_size != 0 and n_bins > 0:
            current_mfcc = current_mfcc[:n_bins * bin_size, :]
            logger.info(f"Truncated MFCC shape to {current_mfcc.shape} for batch {batch_idx}")

        # Init Master Feature Vector (MFV)
        mfv = np.zeros((n_bins * time_domain_features, n_mfcc))
        logger.info(f"MFV shape: {mfv.shape} for batch {batch_idx}")

        # Process each MFCC coefficient
        for i in range(n_mfcc):
            initial = 0
            for j in range(n_bins):
                # Extract bin and compute features
                bin_data = current_mfcc[initial:initial + bin_size, i:i + 1]
                features = compute_time_domain_features(bin_data)

                # Fill MFV, check bounds
                for k in range(time_domain_features):
                    if j * time_domain_features + k < mfv.shape[0]:
                        mfv[j * time_domain_features + k, i] = features[k]
                    else:
                        logger.warning(f"Skipped index {j * time_domain_features + k} for MFCC {i} in batch {batch_idx}")

                initial += bin_size

        mfcct_batch.append(mfv)

    # Stack batches into output
    mfcct_features = np.stack(mfcct_batch, axis=0)
    logger.info(f"Final MFCCT shape: {mfcct_features.shape}")
    return mfcct_features