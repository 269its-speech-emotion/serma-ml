################################################################################
#                                                                              #
#                   This is the features preparation part of the project       #
#                                                                              #
################################################################################

from sklearn.model_selection import train_test_split
from keras.utils import to_categorial 
from typing import Tuple
import numpy as np

from models import cnn_model

def prepare_and_split_data(data: dict, feature_type: str = 'mfcc', val_size: float = 0.2, test_size: float = 0.1) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Splits data into training, validation, and test sets.

    Args:
        data (dict): Dictionary containing features and labels.
        feature_type (str, optional): Feature type to extract (e.g., 'mfcc'). Defaults to 'mfcc'.
        val_size (float, optional): Proportion of validation data. Defaults to 0.2.
        test_size (float, optional): Proportion of test data. Defaults to 0.1.

    Returns:
        Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
            Training, validation, and test features and labels.
    """
    # Extract labels and features
    labels = np.array(data['labels'])
    features = np.array(data['data'][feature_type])

    # Map string labels to integers
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    labels_numeric = np.array([label_to_index[label] for label in labels])

    # One-hot encode the labels
    num_classes = len(unique_labels)
    labels_encoded = to_categorical(labels_numeric, num_classes)

    # Reshape the features for compatibility with CNN models
    height, width = features.shape[1], features.shape[2]
    features = features.reshape(features.shape[0], height, width, 1)


    # Split data into temporary (train+val) and test sets
    x_temp, x_test, y_temp, y_test = train_test_split(
        features, labels_encoded, test_size=test_size,
        stratify=labels_numeric, random_state=42
    )

    # Split temporary data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_size,
        stratify=np.argmax(y_temp, axis=1), random_state=42
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def model_training (x_train, x_val, y_train, y_val):
    
    model = cnn_model()
    
    history = model 
    
    