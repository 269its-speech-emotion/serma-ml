import numpy as np
from from_root import from_root
from sklearn.model_selection import train_test_split

from build_features import build_features, prepare_features_for_cnn
from config import logger

ORGANIZED_DATASET_FOLDER = from_root('data', 'ravdess_organized')


try:
    # Build features
    features = build_features(organized_data_folder=ORGANIZED_DATASET_FOLDER)

    # Prepare the MFCC and MFCCT data for CNN
    mfcc_data, mfcct_data = prepare_features_for_cnn(features)

    # Convert labels to numerical format
    labels = features['labels']
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numerical_labels = np.array([label_to_index[label] for label in labels])

    # Split the data into training and validation sets
    X_train_mfcc, X_val_mfcc, y_train, y_val = train_test_split(mfcc_data, numerical_labels, test_size=0.2,
                                                                random_state=42, stratify=numerical_labels)
    #X_train_mfcct, X_val_mfcct, _, _ = train_test_split(mfcct_data, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels)

    print("Training MFCC data shape:", X_train_mfcc.shape)
    print("Validation MFCC data shape:", X_val_mfcc.shape)
    #print("Training MFCCT data shape:", X_train_mfcct.shape)
    #print("Validation MFCCT data shape:", X_val_mfcct.shape)

    





except Exception as e:
    logger.error(f"Error: {str(e)}")