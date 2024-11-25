from from_root import from_root
from src.data.make_dataset import arrange_dataset_into_emotion_type

RAW_DATASET_FOLDER = from_root('data', 'ravdess_audio_speech_raw')
ARRANGED_DATASET_FOLDER = from_root('data', 'ravdess_audio_speech_processed')

if __name__ == "__main__":
    # First let arrange the folders by speech-emotion type
    arrange_dataset_into_emotion_type(raw_data_folder=RAW_DATASET_FOLDER,
                                      arranged_data_folder=ARRANGED_DATASET_FOLDER)
