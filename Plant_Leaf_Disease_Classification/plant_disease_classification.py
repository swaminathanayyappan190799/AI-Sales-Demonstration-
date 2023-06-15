import os
import cv2
import numpy as np
from log_settings import logger
import tensorflow as tf

class PlantLeadDiseaseClassifier():
    def __init__(self,input_image):
        self.input_img = input_image
        self.label = []
        self.model = None

    def load_classifier_labels(self,labels_path):
        """
        Function that load the model's label from the labels.txt file
        """
        try:
            with open(labels_path,encoding="UTF-8") as label_obj:
                label = [line.strip() for line in label_obj.readlines()]
                logger.info("Classifier labels loaded successfully")
            return label
        except:
            logger.error("Error in load_classifier_labels()")
    
    def load_classifier_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info("Classifier Model loaded successfully")
            return model
        except:
            logger.info("Error in load_classifier_model()")
        

    def run_classifier_infernce(self):
        try:
            self.model = self.load_classifier_model(model_path=f"{os.getcwd()}{os.sep}models{os.sep}plant_leaf_disease_classification.h5")
            self.label = self.load_classifier_labels(labels_path=f"{os.getcwd()}{os.sep}models{os.sep}plant_leaf_disease_classification.txt")
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