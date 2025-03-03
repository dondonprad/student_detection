import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from src.logger import logging
from src.exception import CustomException
from src.components.config import data_config #data_source config 
from src.components.feature_extraction import ImageRepresentation
from src.components.main import ImageRecognation


app = Flask(__name__)


@app.route('/face_verification', methods=['POST'])
def post_example():
    try:
        logging.info('Create API for face detection and save file')

        input_data = request.get_json()
        param1 = input_data['image']
        #img = f"data:image/.jpg;base64,{param1}"
        param2 = input_data['class_id']
        save_path = data_config.database_class + param2
        time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        save_name = f"{save_path}/{param2}_{str(time)}.jpg"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        obj_representation = ImageRepresentation()
        obj_recognation = ImageRecognation()
        res_representation = obj_representation.face_detection_representation(param1)
        comp, nim = obj_representation.calculation_with_img(res_representation, data_config.csv_save_path + 'representation_database.csv')
        res_recognantion = obj_recognation.recognation(param1, comp, nim, save_name)

        response_data = {
            'NIM':res_recognantion
        }

        return jsonify(response_data)

    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    app.run(debug=True)
    



