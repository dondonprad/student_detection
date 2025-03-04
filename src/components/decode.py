import sys
import base64
import cv2
import numpy as np
from src.exception import CustomException
from src.logger import logging

class DecodeImg():
    def decode_img(self, uri):
        try:
            logging.info('decode image process')
            encoded_data = uri.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            # old (python 2 version):
            # nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)

            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        
        except Exception as e:
            raise CustomException(e, sys)
