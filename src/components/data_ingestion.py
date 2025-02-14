import os
import sys

from src.logger import logging
from src.exception import CustomException

from ultralytics import YOLO 
from IPython.display import display, Image
from dataclasses import dataclass 



#import pretrained YOLO Model 
model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model

model.classes = [0]  # 0 is the index for the 'person' class in YOLOv8