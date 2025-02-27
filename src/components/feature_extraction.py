import os
import sys
import numpy as np
import csv
import math
import cv2
import mediapipe as mp
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from src.components.config import data_config
from src.logger import logging
from src.exception import CustomException
from src.components.image_crop import ImageCropConfig
from dataclasses import dataclass 

model_name = "Facenet"
model: FacialRecognition = DeepFace.build_model(task="facial_recognition", model_name=model_name)
target_size = model.input_shape
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

@dataclass
class FeatureExtractionConfig():
   #image_nose_landmark_save_path:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/croped_object_landmark'
   #csv_pose_landmark_save_path:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/csv_landmark'
   mp_pose = mp.solutions.pose
   mp_drawing = mp.solutions.drawing_utils 
   mp_drawing_styles = mp.solutions.drawing_styles
   images = {}
   nose_landmark = {}

@dataclass
class ImageRepresentationConfig():
    obj = data_config.image_data_path
    save_csv =data_config.csv_save_path
    db_path = data_config.database_path

class FeatureExtraction(): #feature extraction for classification student condition
    def __init__(self):
        self.feature_extraction_config = FeatureExtractionConfig()
        self.image_crop_config = ImageCropConfig()
        self.data_config = data_config()

    def images_store(self)->dict: #store image to dictionary
        file_dir = self.data_config.database_path # Specify the directory containing your files
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


    def draw_get_nose_landmark(self, images:dict)->dict: #get landmark value then store on dict
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
                    
                    save_dir = self.data_config.image_landmark_save_path

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = f"{save_dir}/{name}"
                    cv2.imwrite(save_path, annotated_image) # Fixed indentation
                    #self.resize_and_show(annotated_image)

            return self.feature_extraction_config.nose_landmark

        except Exception as e:
           raise CustomException(e,sys)
        
    def csv_pose_landmark_data(self):
        file_dir = self.data_config.image_save_path # Specify the directory containing your files
        

class ImageRepresentation():
    def __init__(self):
        self.obj = ImageRepresentationConfig.obj
        self.save_csv = ImageRepresentationConfig.save_csv
        self.db_path = ImageRepresentationConfig.db_path
    
    def database_representation(self): #create image representation from database then store to csv
        csv_name = self.save_csv + 'representation_database.csv'
        file_dir = self.db_path# Specify the directory containing your files
        file_list = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
        try:
            logging.info('Representation Calculation')
            
            with open(csv_name, mode="w", newline="") as file:
                writer = csv.writer(file)
                header = ['NIM','value']
                writer.writerow(header)

                for file in file_list:
                    row = [f'{os.path.splitext(file)[0]}'] #remove jpg extension string
                    obj = DeepFace.extract_faces(file_dir+file)
                    row.extend([list(np.array(model.forward(np.expand_dims(cv2.resize(obj[0]["face"],target_size), axis=0))))])
                    writer.writerow(row)

        except Exception as e:
            raise CustomException(e, sys)    
        
    def face_detection_representation(self, img)->dict: #read face detection data and store to dict
        try: 
            logging.info('detecting face on image')
            obj = DeepFace.extract_faces(img)
            ig = cv2.imread(img)
            face_representation = {}

            for i in range(len(obj)):
                x,y,w,h,_,_ = obj[i]['facial_area'].values()
                representation = np.array(model.forward(np.expand_dims(cv2.resize(obj[i]["face"],target_size), axis=0)))
                face_representation[str(i)] = [[x,y,w,h],[representation]]

            return face_representation

        except Exception as e:
            raise CustomException(e,sys)
        

    def csv_representation(self, img_representation, db_representation:str)-> list: #read database representation CSV 
        try:
            res = []
            logging.info('load csv database then store to dict')
            with open(db_representation, newline='') as file: 
                    reader = csv.DictReader(file) 
                    for row in reader:
                        res.append(self.distance_calculation(img_representation,np.array(eval(row['value']))))
                        
            return res

        except Exception as e:
            raise CustomException(e,sys)


    def calculation_with_img(self, img_representation:dict, db_representation:str):
        try:
            res_list = []
            nim = []
            threshold = verification.find_threshold(model_name=model_name, distance_metric="euclidean")

            with open(db_representation, newline='') as file: 
                    reader = csv.DictReader(file) 
                    for row in reader:
                        nim.append(row['NIM'])


            for i,j in img_representation.items():
                res = self.csv_representation(j[1], db_representation)
                img_representation[i].append(res)
                res_list.append(res)

            return img_representation, nim
        
        except Exception as e:
            raise CustomException(e,sys)
            
      
    def distance_calculation(self, img_representation, db_img_representation):
        try:
            logging.info('calculate distance using representation')
            img_representation = np.array(img_representation)
            db_img_representation = np.array(db_img_representation)
            distance_vector = np.square(img_representation - db_img_representation)
            current_distance = np.sqrt(distance_vector.sum())
            return current_distance

        except Exception as e:
            raise CustomException(e,sys)



if __name__ == '__main__':
    obj = ImageRepresentation() #.database_representation()
    res= obj.face_detection_representation(ImageRepresentationConfig.obj)
    comp, nim = obj.calculation_with_img(res, ImageRepresentationConfig.save_csv + 'representation_database.csv')
    print(comp)
    print(nim)