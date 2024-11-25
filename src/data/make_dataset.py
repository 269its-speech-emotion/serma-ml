import os
import shutil
from tqdm import tqdm

"""
Each of the 7356 RAVDESS files has a unique filename.
The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4).
The "06" in the third place represents the emotion type and can be as follows:
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
"""
CODE_OF_EMOTIONS = {'01': 'neutral',
                    '02': 'calm',
                    '03': 'happy',
                    '04': 'sad',
                    '05': 'angry',
                    '06': 'fearful',
                    '07': 'disgust',
                    '08': 'surprised'
                    }

def arrange_dataset_into_emotion_type(raw_data_folder, arranged_data_folder) -> None:
    """
    This function rearranges the RAVDESS dataset into folders according to speech-emotion type.
    Args:
        raw_data_folder (str): The path of the original RAVDESS dataset
        arranged_data_folder (str): The path to save the rearranged dataset
    """

    # Create the arranged dataset folder if it doesn't exist
    if not os.path.exists(arranged_data_folder):
        os.makedirs(arranged_data_folder)

    # Loop through each folder in the raw dataset
    for folder in os.listdir(raw_data_folder):
        # Progress bar for tracking files in the folder
        for file in tqdm(os.listdir(os.path.join(raw_data_folder, folder)), desc=f'Arranging {folder}'):

            # Extract the emotion code (3rd part of the filename)
            emotion_code = file[6:8]

            # Check if the code matches any known emotion type
            if emotion_code in CODE_OF_EMOTIONS.keys():
                # Get the corresponding emotion type
                emotion_type = CODE_OF_EMOTIONS[emotion_code]

                # Create a folder for this emotion type if it doesn't exist
                emotion_type_folder = os.path.join(arranged_data_folder, emotion_type)
                emotion_file = os.path.join(raw_data_folder, folder, file)
                if not os.path.exists(emotion_type_folder):
                    os.makedirs(emotion_type_folder)

                # Copy the file to the corresponding emotion folder
                if os.path.isfile(os.path.join(emotion_file)):
                    shutil.copy(src=emotion_file, dst=emotion_type_folder)
            else:
                # Skip files with invalid emotion codes
                continue