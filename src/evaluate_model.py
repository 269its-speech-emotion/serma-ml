import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelBinarizer


def plot_training_history(history, model_type, report_folder):
    """
    Plot training and validation loss and accuracy from the training history.

    :param model_type: Type of model ('cnn', 'lstm', or 'conv1d').
    :param history: The training history object returned by model.fit.
    :param report_folder: folder to save the history plot

    :return: None
    """
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(16, 8))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, 'b', label='Training Loss')
    plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f'Training and Validation Accuracy for {model_type.upper()} model')
    plt.tight_layout()
    plt.savefig(f'{report_folder}/{model_type.upper()}_training_history_plot.png')
    plt.show()


def plot_confusion_matrix(y_true, y_predicted, class_names, model_type, report_folder):
    """
    Plot the confusion matrix for a multi-class classification model.

    :param model_type: Type of model ('cnn', 'lstm', or 'conv1d').
    :param y_true: True labels.
    :param y_predicted: Predicted labels.
    :param class_names: List of class names.
    :param report_folder: folder to save the confusion matrix plot

    :return: None
    """
    cm = confusion_matrix(y_true, y_predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 8))
    plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.title(f'Confusion Matrix for {model_type.upper()} model')
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'{report_folder}/{model_type.upper()}_confusion_matrix.png')
    plt.show()


def plot_roc_auc_curve(model, x_test, y_test, class_names, model_type, report_folder):
    """
    Plot the ROC curve and calculate the AUC for a multi-class classification model.

    :param model_type: Type of model ('cnn', 'lstm', or 'conv1d').
    :param model: The trained classification model.
    :param x_test: Test data.
    :param y_test: True labels for the test data (one-hot encoded).
    :param class_names: List of class names.
    :param report_folder: Folder to save the ROC Curve

    :return: None
    """

    n_classes = len(class_names)
    plt.figure(figsize=(16, 8))

    label_binarizer = LabelBinarizer()
    y_test = label_binarizer.fit_transform(y_test)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], model.predict(x_test)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (class {i}, area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Receiver Operating Characteristic (ROC) for {model_type.upper()} model")
    plt.legend(loc='lower right')
    plt.savefig(f'{report_folder}/{model_type.upper()}_ROC_AUC_curve.png')
    plt.show()


def model_performance_and_assessment(history, model, x_test, y_test, class_names, model_type, report_folder):
    """
    This function shows the performance of a model during training and testing.
    Plot training and validation loss/accuracy graphs.
    Save the classification Report to a csv file.

    :param history: Training history of the model.
    :param model_type: Type of model ('cnn', 'lstm', or 'conv1d').
    :param model: The trained classification model.
    :param x_test: Test data.
    :param y_test: True labels for the test data (one-hot encoded).
    :param class_names: List of class names.
    :param report_folder: Folder to save reporting data

    :return: None
    """

    # Let's plot the history of the model to see the performance during training and validation.
    plot_training_history(history=history, model_type=model_type, report_folder=report_folder)

    # Predict class labels
    predicted_y = np.argmax(model.predict(x_test), axis=1)

    # Ensure that y_test is a 1D array of integers
    y_test = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Calculate and display accuracy, F1-score, precision, recall
    accuracy = np.mean(predicted_y == y_test)
    report = classification_report(y_test, predicted_y, target_names=class_names)
    print(f"{model_type.upper()} Accuracy:", accuracy)
    print(f"\n{model_type.upper()} Classification Report:\n", report)

    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(
        classification_report(y_test, predicted_y, target_names=class_names, output_dict=True)).transpose().round(2)

    # Save the DataFrame to a CSV file
    report_df.to_csv(f'{report_folder}/{model_type.upper()}_classification_report.csv', index=True)
    print(f"{model_type.upper()} Classification Report created and saved successfully.")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, predicted_y, class_names, model_type=model_type, report_folder=report_folder)
    print(f"{model_type.upper()} Confusion Matrix created and saved successfully.")

    # Plot ROC curve and AUC
    plot_roc_auc_curve(model, x_test, y_test, class_names, model_type=model_type, report_folder=report_folder)
    print(f"{model_type.upper()} ROC curve and AUC created and saved successfully.")