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

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

@dataclass
class FeatureExtractionConfig():
   image_nose_landmark_save_path:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/croped_object_landmark'
   csv_pose_landmark_save_path:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/csv_landmark'
   mp_pose = mp.solutions.pose
   mp_drawing = mp.solutions.drawing_utils 
   mp_drawing_styles = mp.solutions.drawing_styles
   images = {}
   nose_landmark = {}

class FeatureExtraction():
    def __init__(self):
        self.feature_extraction_config = FeatureExtractionConfig()
        self.image_crop_config = ImageCropConfig()

    def images_store(self)->dict: #store image to dictionary
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
            
            return self.feature_extraction_config.images

        except Exception as e:
           raise CustomException(e,sys)



    def resize_and_show(image):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
        #cv2.imshow(img)


    def draw_get_nose_landmark(self, images:dict)->dict: #get landmark value then store at dict
        # Run MediaPipe Pose and draw pose landmarks.
        try:
            with self.feature_extraction_config.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
                for name, image in images.items():
                    # Convert the BGR image to RGB and process it with MediaPipe Pose.
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # Print nose landmark.
                    image_hight, image_width, _ = image.shape
                    if not results.pose_landmarks:
                        continue

                    '''
                    print(
                            f'Nose coordinates: ('
                            f'{results.pose_landmarks.landmark[self.feature_extraction_config.mp_pose.PoseLandmark.NOSE].x * image_width}, '
                            f'{results.pose_landmarks.landmark[self.feature_extraction_config.mp_pose.PoseLandmark.NOSE].y * image_hight})'
                         )
                    '''
                    self.feature_extraction_config.nose_landmark[name] = [results.pose_landmarks.landmark[self.feature_extraction_config.mp_pose.PoseLandmark.NOSE].x * image_width,
                                                                     results.pose_landmarks.landmark[self.feature_extraction_config.mp_pose.PoseLandmark.NOSE].y * image_hight]
                    

                    # Draw pose landmarks.
                    print(f'Pose landmarks of {name}:')
                    annotated_image = image.copy()
                    self.feature_extraction_config.mp_drawing.draw_landmarks( annotated_image, results.pose_landmarks,
                                               self.feature_extraction_config.mp_pose.POSE_CONNECTIONS,
                                               landmark_drawing_spec = self.feature_extraction_config.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    save_dir = self.feature_extraction_config.image_nose_landmark_save_path

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = f"{save_dir}/{name}"
                    cv2.imwrite(save_path, annotated_image) # Fixed indentation
                    #self.resize_and_show(annotated_image)

            return self.feature_extraction_config.nose_landmark

        except Exception as e:
           raise CustomException(e,sys)
        
    def csv_pose_landmark_data(self):
        file_dir = self.image_crop_config.image_save_path # Specify the directory containing your files
        


if __name__ == '__main__':
    obj = FeatureExtraction()
    res = obj.images_store()
    land = obj.draw_get_nose_landmark(res)
    print(land)