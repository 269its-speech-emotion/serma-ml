

import matplotlib.pyplot as plt
import numpy as np

from src.config import logger


def plot_signals(original_signal, preprocessed_signal, sample_rate=16000):
    """Plot original and preprocessed audio signals for comparison.

    Args:
        original_signal (np.ndarray): Raw audio data (shape: n_samples).
        preprocessed_signal (np.ndarray): Processed audio data (shape: n_samples).
        sample_rate (int): Audio sample rate (default: 16000 Hz).

    Returns:
        None
    """
    # Compute time axes in seconds
    time_original = np.linspace(0, len(original_signal) / sample_rate, len(original_signal))
    time_preprocessed = np.linspace(0, len(preprocessed_signal) / sample_rate, len(preprocessed_signal))

    # Create plot with two subplots
    plt.figure(figsize=(12, 6))

    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(time_original, original_signal, color='blue')
    plt.title('Original Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot preprocessed signal
    plt.subplot(2, 1, 2)
    plt.plot(time_preprocessed, preprocessed_signal, color='red')
    plt.title('Preprocessed Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    logger.info("Plotted original vs. preprocessed signals")