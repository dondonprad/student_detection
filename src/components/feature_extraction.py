import os
import sys
import numpy as np
import math
import cv2
import mediapipe as mp

from src.logger import logging
from src.exception import CustomException
from src.components.image_crop import ImageCropConfig
from dataclasses import dataclass 

#DESIRED_HEIGHT = 480
#DESIRED_WIDTH = 480

@dataclass
class FeatureExtractionConfig():
   mp_pose = mp.solutions.pose
   mp_drawing = mp.solutions.drawing_utils 
   mp_drawing_styles = mp.solutions.drawing_styles
   images = {}
   landmark = {}

class FeatureExtraction():
    def __init__(self):
        self.feature_extraction_config = FeatureExtractionConfig()
        self.image_crop_config = ImageCropConfig()

    def images_store(self): #store image to dictionary
        file_dir = self.image_crop_config.image_save_path # Specify the directory containing your files
        file_list = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))] # Get a list of all files in the directory
        try:
            logging.info('store images to dictionary')
            for filename in file_list:
                file_path = os.path.join(file_dir, filename)
                print(f"Processing {filename}...")
                # Read images with OpenCV and store in the dictionary
                image = cv2.imread(file_path)  # Read the image using file_path
                if image is not None:  # Check if image was loaded successfully
                    self.feature_extraction_config.images[filename] = image #input images to dict
                else: 
                    print(f"Could not load image: {file_path}")
            
            #return self.feature_extraction_config.images.items()

        except Exception as e:
           raise CustomException(e,sys)



    def resize(self, image):
        pass


    def draw_get_landmark(self):
        # Run MediaPipe Pose and draw pose landmarks.
        try:
            with self.feature_extraction_config.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
                for name, image in self.feature_extraction_config.images.items():
                    # Convert the BGR image to RGB and process it with MediaPipe Pose.
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # Print nose landmark.
                    image_hight, image_width, _ = image.shape
                    if not results.pose_landmarks:
                        continue
                    print(
                            f'Nose coordinates: ('
                            f'{results.pose_landmarks.landmark[self.feature_extraction_config.mp_pose.PoseLandmark.NOSE].x * image_width}, '
                            f'{results.pose_landmarks.landmark[self.feature_extraction_config.mp_pose.PoseLandmark.NOSE].y * image_hight})'
                         )

        except:
            pass


if __name__ == '__main__':
    obj = FeatureExtraction()
    res = obj.images_store()
    print(res)