from from_root import from_root
from src.data.make_dataset import organize_dataset_into_emotion_type, create_production_data

RAW_DATASET_FOLDER = from_root('data', 'ravdess_raw')
ORGANIZED_DATASET_FOLDER = from_root('data', 'ravdess_organized')
PRODUCTION_DATASET_FOLDER = from_root('data', 'ravdess_production')

if __name__ == "__main__":
    # First let arrange the folders by speech-emotion type
    print("Let arrange the folders by speech-emotion type")
    organize_dataset_into_emotion_type(origin_data_folder=RAW_DATASET_FOLDER,
                                       organized_data_folder=ORGANIZED_DATASET_FOLDER)

    # Prepare the production data
    print("Prepare the production data")
    create_production_data(processed_data_folder=ORGANIZED_DATASET_FOLDER,
                           production_data_folder=PRODUCTION_DATASET_FOLDER,
                           n_sample=5)




