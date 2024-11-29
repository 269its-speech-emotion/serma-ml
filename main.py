"""from from_root import from_root
from src.make_dataset import organize_dataset_into_emotion_type, create_production_data
from src.visualize import get_random_audio_files, plot_audio_features
from src.build_features import extract_audio_features
from src.models.train_model import prepare_and_split_data


RAW_DATASET_FOLDER = from_root('data', 'ravdess_raw')
ORGANIZED_DATASET_FOLDER = from_root('data', 'ravdess_organized')
PRODUCTION_DATASET_FOLDER = from_root('data', 'ravdess_production')

if __name__ == "__main__":
    # First let arrange the folders by speech-emotion type
    print("Let arrange the folders by speech-emotion type")
    #organize_dataset_into_emotion_type(origin_data_folder=RAW_DATASET_FOLDER,
    #                                   organized_data_folder=ORGANIZED_DATASET_FOLDER)

    # Prepare the production data
    print("Prepare the production data")
    #create_production_data(processed_data_folder=ORGANIZED_DATASET_FOLDER,
    #                       production_data_folder=PRODUCTION_DATASET_FOLDER,
    #                       n_sample=5)

    # Get random audio files
    #print("Get random audio files")
    #random_audio_samples = get_random_audio_files(organized_data_folder=ORGANIZED_DATASET_FOLDER,
    #                                          n_sample=6)
    # Plot the random audio files
    #print("Make the plot of the audio characteristics.")
    #plot_audio_features(audio_samples=random_audio_samples, n_samples=6)

    dataset = extract_audio_features(organized_data_folder=ORGANIZED_DATASET_FOLDER)

    x_train, x_val, x_test, y_train, y_val, y_test = prepare_and_split_data(data=dataset)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)
    """
import tensorflow as tf # type: ignore
import keras # type: ignore

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
