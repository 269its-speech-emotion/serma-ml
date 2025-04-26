import numpy as np
from from_root import from_root
from sklearn.model_selection import train_test_split

from src.build_features import build_features, prepare_features_for_cnn
from src.config import logger
from src.make_dataset import organize_dataset_into_emotion_type, preprocess_and_save_dataset
from src.models import cnn_model
import tensorflow as tf
from build_features import build_ml_classifiers

ORGANIZED_DATASET_FOLDER = from_root('data', 'ravdess_organized')
ORIGIN_DATASET_FOLDER = from_root('data', 'ravdess_raw')
PREPROCESSED_DATASET_FOLDER = from_root('data', 'ravdess_preprocessed')

try:
    # Arrange and preprocess data (uncomment if needed)
    # organize_dataset_into_emotion_type(ORIGIN_DATASET_FOLDER, ORGANIZED_DATASET_FOLDER)
    # preprocess_and_save_dataset(ORGANIZED_DATASET_FOLDER, PREPROCESSED_DATASET_FOLDER)

    # Build features
    features_data = build_features(PREPROCESSED_DATASET_FOLDER)
    logger.info(f"Built features for {len(features_data['labels'])} samples")

    # Prepare features for CNN and ML
    mfcc_data, mfcct_data = prepare_features_for_cnn(features_data)
    logger.info(f"Prepared MFCC shape: {mfcc_data.shape}, MFCCT shape: {mfcct_data.shape}")

    # Convert labels to numerical format
    labels = np.array(features_data['labels'])
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numerical_labels = np.array([label_to_index[label] for label in labels])

    # Train and evaluate ML classifiers on MFCCT
    #ml_results = build_ml_classifiers(mfcct_data, numerical_labels)
    #for name, result in ml_results.items():
    #    logger.info(f"{name} Results - Accuracy: {result['accuracy']:.4f}, CV Accuracy: {result['cv_mean_accuracy']:.4f}")

    # Optionally, train CNN on MFCCT (as before, with adjusted input shape)
    input_shape = mfcc_data.shape[1:]  # e.g., (48, 13) after padding
    model_mfcc = cnn_model(input_shape=input_shape, num_classes=len(unique_labels))
    history = model_mfcc.fit(mfcc_data, numerical_labels,  # Use full dataset or split for CNN
                              validation_split=0.2, epochs=100, batch_size=16, verbose=1)
    logger.info("CNN training completed successfully")

except Exception as e:
    logger.error(f"Error in pipeline: {str(e)}", exc_info=True)