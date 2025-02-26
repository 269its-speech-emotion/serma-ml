################################################################################
#                                                                              #
#                    Audio Preprocessing Component for SER Project             #
#                   Prepares audio data by removing silence,                   #
#                   applying pre-emphasis, and normalizing signals.            #
#                                                                              #
################################################################################

import librosa
import numpy as np
from scipy.signal import lfilter

from config import logger


def preprocess_audio_file(audio_path, sample_rate=16000, preemphasis_alpha=0.97, min_length=1600):
    """Preprocess audio for SER: remove silence, apply pre-emphasis, normalize.

    Args:
        audio_path (str): Path to audio file (e.g., WAV).
        sample_rate (int): Target sample rate (default: 16000 Hz).
        preemphasis_alpha (float): Pre-emphasis coefficient (default: 0.97).
        min_length (int): Min samples after preprocessing (default: 1600, ~0.1s).

    Returns:
        np.ndarray: Preprocessed audio (shape: n_samples).
    """
    # Load audio, preserving original rate
    signal, sr = librosa.load(audio_path, sr=None)
    logger.info(f"Loaded signal: length={len(signal)}, rate={sr} Hz")

    # Resample to target rate if needed
    if sr != sample_rate:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=sample_rate)
        logger.info(f"Resampled to {sample_rate} Hz, length={len(signal)}")

    # Remove silence with lenient threshold
    signal, _ = librosa.effects.trim(signal, top_db=10)
    logger.info(f"After silence removal, length={len(signal)}")

    # Ensure sufficient length
    if len(signal) < min_length:
        raise ValueError(f"Signal too short: {len(signal)} < {min_length} samples")

    # Apply pre-emphasis filter
    preemphasized_signal = lfilter([1, -preemphasis_alpha], [1], signal)
    logger.info(f"Applied pre-emphasis, alpha={preemphasis_alpha}")

    # Normalize signal
    mean = np.mean(preemphasized_signal)
    std = np.std(preemphasized_signal)

    if std == 0:
        std = 1e-10
        logger.warning("Std zero, set to 1e-10 to avoid division")

    normalized_signal = (preemphasized_signal - mean) / std
    logger.info(f"Normalized, mean={mean:.4f}, std={std:.4f}")

    logger.info(f"---------Audio file {audio_path.name} was preprocessed successfully---------")
    return normalized_signal