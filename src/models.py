import keras


def cnn_model(input_shape, num_classes, learning_rate=0.0001):

    model = keras.models.Sequential()

    # First Convolutional Layer
    model.add(keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, input_shape=input_shape, activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=4))

    # Second Convolutional Layer
    model.add(keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=4))

    # Third Convolutional Layer
    model.add(keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())

    # Fully Connected Layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model