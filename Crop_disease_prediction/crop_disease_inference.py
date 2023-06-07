import os
import cv2
import numpy as np
from logger import logger
import tensorflow as tf

class CropDiseasePrediction():
    def __init__(self,input_image):
        self.input_img = input_image
        self.label = []
        self.model = None
    
    def load_classifier_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info("Classifier Model loaded successfully")
            return model
        except:
            logger.info("Error in load_classifier_model()")
        

    def run_classifier_infernce(self):
        try:
            self.model = self.load_classifier_model(model_path=f"{os.getcwd()}{os.sep}models{os.sep}crop_disease_classifier.h5")
            self.label = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            # cv2_img = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2BGR)
            new_img = cv2.resize(self.input_img, (224, 224))
            # grayscale_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            # normalized_image = grayscale_image / 255.0
            # input_image = np.expand_dims(normalized_image, axis=-1)
            input_image = np.expand_dims(new_img, axis=0)
            # processed_img = np.expand_dims(new_img, axis=0)
            processed_img = np.float32(input_image / 255) #Major step to comvert images to float32 type which generates different classification confidence levels
            result = self.model.predict(processed_img)
            output_list = result[0].tolist()
            return self.label[output_list.index(max(output_list))]
        except:
            logger.error("Error in run_classifier_inference()")