################################################################################
#                                                                              #
#                   This is the features preparation part of the project       #
#                                                                              #
################################################################################

from typing import Tuple

import keras

'''
def cnn_model(input_shape: Tuple[int, int, int], num_classes: int = 8) -> models.Sequential:
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
'''

def cnn_1d_model(input_shape: Tuple[int, int, int], num_classes: int = 8) -> keras.models.Sequential :

    model = keras.models.Sequential()

    # Conv_1 layer
    model.add(keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', input_shape=input_shape))
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Conv_2 layer
    model.add(keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same'))
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Conv_3 layer
    model.add(keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same'))
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))

    # Flatten layer
    model.add(keras.layers.Flatten())

    # Fully Connected Layer (Dense)
    model.add(keras.layers.Dense(units=num_classes))
    model.add(keras.layers.Softmax())

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model


