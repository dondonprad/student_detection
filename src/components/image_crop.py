import os
import sys
import string
import cv2

from src.logger import logging
from src.exception import CustomException

from ultralytics import YOLO 
from IPython.display import display, Image
from dataclasses import dataclass 


@dataclass
class ImageCropConfig():

    image_data_path:str = '/home/server-iss-mbkm/project/student_detection_dataset/test.jpg'
    image_save_path:str = '/home/server-iss-mbkm/project/student_detection_dataset/croped_object'

    person_yolo_model = YOLO('yolov8n.pt') # Load a pretrained YOLOv8n model
    person_yolo_model.classes = [0] # 0 is the index for the 'person' class in YOLOv8


class ImageCrop():
    def __init__ (self):
        self.image_crop_config = ImageCropConfig()

    def initiate_crop_image(self):
        "This Function is Responsible for cropping image"
        logging.info('Initiating Crop Object')
        try:
            
            image = cv2.imread(self.image_crop_config.image_data_path)
            results = self.image_crop_config.person_yolo_model(image)
            save_path = self.image_crop_config.image_save_path

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for i, r in enumerate(results[0].boxes.data):
                # Ambil koordinat bounding box
                x1, y1, x2, y2, conf, cls = r
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop gambar
                cropped = image[y1:y2, x1:x2]

                # Dapatkan nama class
                class_name = self.image_crop_config.person_yolo_model.names[int(cls)]

                # Simpan gambar hasil crop
                if(class_name == "person"):
                    save_name = f"{save_path}/{class_name}_{i}.jpg"
                    cv2.imwrite(save_name, cropped)

                print(f"Saved {class_name} crop to {save_path}")

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = ImageCrop()
    obj.initiate_crop_image()
    


