import shutil
from tqdm import tqdm
from pathlib import  Path
import random

"""
Each of the 7356 RAVDESS files has a unique filename.
The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4).
The "06" in the third place represents the emotion type and can be as follows:
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
"""
# Define a dictionary mapping emotion codes to emotion names
CODE_OF_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def organize_dataset_into_emotion_type(origin_data_folder: Path, organized_data_folder: Path) -> None:
    """
    Rearranges the RAVDESS dataset into subfolders based on emotion types.
    Args:
        origin_data_folder (Path): Path to the original RAVDESS dataset directory.
        organized_data_folder (Path): Path to the target directory for the organized dataset.
    """
    # Ensure the arranged data folder exists
    organized_data_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through each actor's folder in the raw dataset
    for folder in origin_data_folder.iterdir():
        if folder.is_dir():  # Ensure it's a directory
            # Progress bar for files in the current folder
            for file in tqdm(folder.iterdir(), desc=f'Arranging {folder.name}'):
                # Extract the emotion code from the filename (positions 6 and 7)
                emotion_code = file.name[6:8]

                # Check if the emotion code is valid
                if emotion_code in CODE_OF_EMOTIONS:
                    # Map the emotion code to the corresponding emotion name
                    emotion_type = CODE_OF_EMOTIONS[emotion_code]
                    emotion_type_folder = organized_data_folder / emotion_type

                    # Create the emotion-specific folder if it doesn't exist
                    emotion_type_folder.mkdir(parents=True, exist_ok=True)

                    # Copy the file to the appropriate emotion folder
                    if file.is_file():
                        shutil.copy(src=file, dst=emotion_type_folder)


def create_production_data(processed_data_folder: Path, production_data_folder: Path, n_sample: int) -> None:
    """
    Moves random audio files from each emotion folder into the production folder.
    Args:
        processed_data_folder (Path): Folder with processed data organized by emotion.
        production_data_folder (Path): Folder to store the production dataset.
        n_sample (int): Number of audio files to move per emotion.
    """
    # Create the production folder if it doesn't exist
    production_data_folder.mkdir(parents=True, exist_ok=True)

    # Loop through each emotion folder
    for emotion_folder in processed_data_folder.iterdir():
        if emotion_folder.is_dir():  # Process only directories
            # List all audio files in the current folder
            audio_files = list(emotion_folder.glob("*"))

            # Randomly select files
            selected_audio_files = random.sample(audio_files, min(n_sample, len(audio_files)))

            # Move and rename the selected files
            for file_path in selected_audio_files:
                emotion_code = CODE_OF_EMOTIONS[file_path.stem[6:8]]  # Extract emotion type
                new_file_name = f"{emotion_code}_{file_path.name}"
                shutil.move(str(file_path), str(production_data_folder / new_file_name))