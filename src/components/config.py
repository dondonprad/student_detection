import os
import sys
from dataclasses import dataclass

DATA_SOURCE_PATH:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/raw_object/test.jpg'
IMAGE_SAVE_PATH:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/croped_object'
IMAGE_FEATURE_LANDMARK_PATH:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/croped_object_landmark'
FEATURE_EXTRACTION_CSV_PATH:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/csv_landmark'
OBJ_DATABASE_PATH:str = '/home/server-iss-mbkm/project/project_student_detection/student_detection_dataset/foto_wajah_nama_db'

@dataclass
class data_config():
    image_data_path:str = DATA_SOURCE_PATH
    image_save_path:str = IMAGE_SAVE_PATH
    image_landmark_save_path:str = IMAGE_FEATURE_LANDMARK_PATH
    csv_landmark_save_path:str = FEATURE_EXTRACTION_CSV_PATH
    database_path:str = OBJ_DATABASE_PATH