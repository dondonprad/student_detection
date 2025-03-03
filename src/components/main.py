import sys
import cv2
import base64
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.feature_extraction import ImageRepresentation, ImageRepresentationConfig
from src.components.config import data_config


class ImageRecognation():
    def uri_to_cv2(self, uri_input):
        #encoded_data = uri.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(uri_input.split(',')[1], np.uint8))
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def recognation(self, uri, img_representation:dict, nim_list:list, save_file:str):
        try:
            npm = []
            obj = img_representation
            img = self.uri_to_cv2(uri)
            ig = cv2.imread(img)
            logging.info('Face Recognantion')
            
            for i in range(len(obj)):
                x,y,w,h = obj[str(i)][0][0], obj[str(i)][0][1], obj[str(i)][0][2], obj[str(i)][0][3]
                cv2.rectangle(ig, (x, y), (x + w, y + h), (0, 255, 0), 1)

                tmp = obj[str(i)][2]
                idx = tmp.index(min(tmp))
                cv2.putText(ig,nim_list[idx],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                npm.append(nim_list[idx])

            cv2.imwrite(save_file, ig)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows() 
            return npm
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
if __name__ == '__main__':
    obj = ImageRepresentation() 
    test = ImageRecognation()
    res= obj.face_detection_representation(data_config.image_data_path)
    comp, nim = obj.calculation_with_img(res, data_config.csv_save_path + 'representation_database.csv')
    res_2 = test.recognation(comp, nim, 'cropped')


