import os
import sys
import cv2
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.feature_extraction import ImageRepresentation, ImageRepresentationConfig
from deepface import DeepFace
from src.components.config import data_config


class ImageRecognation():
    def __init__(self):
        pass #self.calculation = ImageRepresentation.calculation_with_img(self)

    def recognation(self, img_representation:dict, nim_list:list):
        try:
            npm = []
            obj = img_representation
            ig = cv2.imread(ImageRepresentationConfig.obj)
            logging.info('Face Recognantion')
            
            for i in range(len(obj)):
                x,y,w,h = obj[str(i)][0][0], obj[str(i)][0][1], obj[str(i)][0][2], obj[str(i)][0][3]
                cv2.rectangle(ig, (x, y), (x + w, y + h), (0, 255, 0), 1)

                tmp = obj[str(i)][2]
                idx = tmp.index(min(tmp))
                cv2.putText(ig,nim[idx],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                npm.append(nim[idx])

            cv2.imshow('Cropped', ig)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
            return npm
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
if __name__ == '__main__':
    obj = ImageRepresentation() 
    test = ImageRecognation()
    res= obj.face_detection_representation(data_config.image_data_path)
    comp, nim = obj.calculation_with_img(res, data_config.csv_save_path + 'representation_database.csv')
    res_2 = test.recognation(comp, nim)


