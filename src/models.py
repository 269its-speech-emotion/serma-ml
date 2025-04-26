import tensorflow as tf

def cnn_model(input_shape, num_classes, learning_rate=0.0001):
    """Build CNN for SER using MFCCT features, per the paper's Section 4.

    Args:
        input_shape (tuple): Shape of MFCCT input (e.g., (12, 13)).
        num_classes (int): Number of emotion classes (e.g., 8).
        learning_rate (float): Adam optimizer learning rate (default: 0.0001).

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = tf.keras.models.Sequential()

    # First Conv1D layer with pooling
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, input_shape=input_shape,
                                    activation='relu', padding='same', name='conv1d_1'))
    model.add(tf.keras.layers.Dropout(0.2, name='dropout_1'))
    model.add(tf.keras.layers.BatchNormalization(name='batchnorm_1'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpool_1'))

    # Second Conv1D layer with pooling
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu',
                                    padding='same', name='conv1d_2'))
    model.add(tf.keras.layers.Dropout(0.2, name='dropout_2'))
    model.add(tf.keras.layers.BatchNormalization(name='batchnorm_2'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpool_2'))

    # Third Conv1D layer with pooling
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu',
                                    padding='same', name='conv1d_3'))
    model.add(tf.keras.layers.Dropout(0.2, name='dropout_3'))
    model.add(tf.keras.layers.BatchNormalization(name='batchnorm_3'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpool_3'))

    # Flattening
    model.add(tf.keras.layers.Flatten(name='flatten'))

    # Dense layer for classification
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_output'))

    model.summary()

    # Compile with Adam and sparse categorical cross-entropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model