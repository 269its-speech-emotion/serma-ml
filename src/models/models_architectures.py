################################################################################
#                                                                              #
#                   This is the features preparation part of the project       #
#                                                                              #
################################################################################

from typing import Tuple
from keras.models import Sequential
from keras.layers import (Conv2D, LSTM, Conv1D, MaxPooling1D, MaxPool2D, Flatten, Dense, Dropout,
                          TimeDistributed, InputLayer, Concatenate, Reshape)


def cnn_model(input_shape: Tuple[int, int, int], num_classes: int = 8) -> Sequential:
    """
    Building a 2D CNN model.
    Args:
        input_shape (Tuple[int, int, int]): The shape of the input audio features.
        num_classes (int): The number of classes to classify (8)
    Returns:
        Sequential: The CNN model
    """
    model = Sequential()

    # Block 1: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 3: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Flatten and add fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
