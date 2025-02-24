import os
import sys
import cv2
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components import config
from src.components.feature_extraction import FeatureExtraction
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from src.components import config

model_name = "Facenet"
model: FacialRecognition = DeepFace.build_model(task="facial_recognition", model_name=model_name)
target_size = model.input_shape


class ImageRecognation():
    def __init__(self):
        self.obj = config.data_config.image_data_path
        self.save_csv = config.data_config.csv_landmark_save_path
        self.db_path = config.data_config.database_path
        self.image_store = FeatureExtraction()

    def recognation(self):
        try:
            obj = DeepFace.extract_faces(self.obj)
            ig = cv2.imread(self.obj)
            representation = []
            logging.info('Face Recognantion')
            #threshold = verification.find_threshold(model_name=model_name, distance_metric="euclidean")
            for i in range(len(obj)):
                representation.append(np.sqrt(np.square(np.array(model.forward(np.expand_dims(cv2.resize(obj[i]["face"],target_size), axis=0))))))
                x,y,w,h,_,_ = obj[i]['facial_area'].values()
                cv2.rectangle(ig, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv2.imshow('Cropped', ig)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 

        except Exception as e:
            raise CustomException(e,sys)
        
    def database_calculation(self):
        
        #tmp1 = [] #image from detection
        #tmp2 = [] #image from database
        #db_path:str = self.db_path #database path
        #file_list = [f for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))] #list file in data_base and as npm 
        csv_save_name = self.save_csv + 'representation_database.csv'

        try:
            logging.info('Similarity Calculation')
            store = self.image_store.images_store()
            nim = []
            value = []
            

            # distance between two images - euclidean distance formula
            for j, k in store.items():
                value.append((np.sqrt(np.square(np.array(model.forward(np.expand_dims(cv2.resize(store[j], target_size), axis=0)))).sum())))
                nim.append()

            return data
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = ImageRecognation().database_calculation()
    print(obj)
    #for person, dist in obj.items():
        #print([person, dist])

