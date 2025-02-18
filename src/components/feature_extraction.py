import os
import sys
import numpy as np
import math
import cv2
import mediapipe as mp

'''
sys.path.append('home/server-iss-mbkm/project/student_detection/src')
sys.path.append('home/server-iss-mbkm/project/student_detection/src')
sys.path.append('home/server-iss-mbkm/project/student_detection/src/components')
'''

from src.logger import logging
from src.exception import CustomException
from src.components.image_crop import ImageCropConfig

from dataclasses import dataclass 

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


@dataclass
class FeatureExtractionConfig():
   mp_pose = mp.solutions.pose
   mp_drawing = mp.solutions.drawing_utils 
   mp_drawing_styles = mp.solutions.drawing_styles
   images = {}

class FeatureExtraction():
    def __init__(self):
        self.feature_extraction_config = FeatureExtractionConfig()
        self.image_crop_config = image_crop.ImageCropConfig()

    def images_store(self): #store image to dictionary
        file_dir = self.image_crop_config #.image_save_path # Specify the directory containing your files
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
            
            return self.feature_extraction_config.images.items()

        except Exception as e:
           raise CustomException(e,sys)



    def resize(self, image):
        pass


if __name__ == '__main__':
    obj = FeatureExtraction()
    res = obj.images_store()
    print(res)