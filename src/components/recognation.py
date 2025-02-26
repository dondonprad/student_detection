import os
import sys
import cv2
import csv
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.config import data_config
from deepface import DeepFace
from deepface.modules import verification
from src.components import config




class ImageRecognation():
    def __init__(self):
        self.obj = config.data_config.image_data_path
        self.save_csv = config.data_config.csv_save_path
        self.db_path = data_config.database_path

    def recognation(self):
        try:
            obj = DeepFace.extract_faces(self.obj)
            ig = cv2.imread(self.obj)
            logging.info('Face Recognantion')
            
            for i in range(len(obj)):
                x,y,w,h,_,_ = obj[i]['facial_area'].values()
                cv2.rectangle(ig, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow('Cropped', ig)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 

        except Exception as e:
            raise CustomException(e,sys)
        
    


if __name__ == '__main__':
    #obj = ImageRecognation().database_calculation()
    obj = ImageRecognation().recognation()
    print(obj)

