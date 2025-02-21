import sys
import cv2
import numpy as np
from src.logger import logging
from src.exception import CustomException
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from src.components import config

class ImageRecognation():
    def __init__(self):
        self.obj = config.data_config.image_data_path

    def recognantion(self):
        obj = DeepFace.extract_faces(self.obj)
        ig = cv2.imread(self.obj)
        try:
            logging.info('Face Recognantion')
            for i in range(len(obj)):
                x,y,w,h,_,_ = obj[i]['facial_area'].values()
                cv2.rectangle(ig, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv2.imshow('Cropped', ig)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = ImageRecognation().recognantion()

