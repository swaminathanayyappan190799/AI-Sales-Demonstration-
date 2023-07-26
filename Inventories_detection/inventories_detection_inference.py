import os
import cv2
import random,string
from ultralytics import YOLO
from log_settings import logger


class InventoryDetection():
    def __init__(self, input_image) -> None:
        self.input_data = input_image
        self.model = None
        self.labels = None
        logger.info("Initating Inventories Detection for the input images")
    
    def load_model(self,model_path):
        try:
            model = YOLO(model=model_path)
            return model
        except:
            logger.error("Error in loading Inventories Model")

    def load_labels(self,model_file):
        try:
            labels = list(model_file.names.values())
            return labels
        except:
            logger.error("Error in loading Inventories model labels")
    
    def save_image(self, img_array):
        try:
            img_path = f"{os.getcwd()}{os.sep}detections{os.sep}{''.join(random.choices(string.ascii_letters, k=5))}.jpg"
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename=img_path,img=img_array)
            return img_path
        except:
            logger.error("Error in saving img !")

    def perform_inference(self, filepath, model):
        """
        Function to perform inference using the image stored in tempdir
        """
        try:
            logger.info("Starting Inference using YOLO")
            results = model.predict(source=filepath, conf=0.25)
            bb_coords=results[0].boxes.xyxy.tolist()
            confidence=results[0].boxes.conf.tolist()
            classes=results[0].boxes.cls.tolist()
            logger.info("Inference completed sending results for drawing annotations")
            return bb_coords, confidence, classes
        except:
            logger.info("Error in perform_inference()")

    def draw_annotations_on_image(self, image_path, inference_data):
        """
        Function to draw annotations using the inference data.
        """
        try:
            img = cv2.imread(image_path)
            color_scheme = {"InventoryBox":(184, 48, 48),'label':(255,255,255)}
            counts = {"InventoryBox":0}
            for val in zip(inference_data[0],inference_data[1],inference_data[2]):
                x,y,w,h = val[0]
                confidence = str(round(val[1]*100,2))
                classes = self.labels[int(val[2])]

                if classes == "InventoryBox":
                    counts["InventoryBox"]+=1

                #For drawing bounding box
                cv2.rectangle(
                    img=img, 
                    pt1=(int(x),int(y)),
                    pt2=(int(w),int(h)), 
                    color=color_scheme[classes], 
                    thickness=2
                )
                
                #Rectangle for displaying class of a detected object
                start_point = (int(x), int(y)-25)
                end_point = (int(w), int(y))
                cv2.rectangle(
                    img=img,
                    pt1=start_point,
                    pt2=end_point,
                    color=color_scheme[classes],
                    thickness=-1
                )

                # Calculate text size and font scale
                (text_width, text_height), _ = cv2.getTextSize(f"{classes}  {confidence}%", cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4)
                font_scale = min((end_point[0] - start_point[0]) / text_width, (end_point[1] - start_point[1]) / text_height)
                # Calculate text location
                text_x = int(start_point[0] + (end_point[0] - start_point[0] - text_width * font_scale) / 2)
                text_y = int(start_point[1] + (end_point[1] - start_point[1] + text_height * font_scale) / 2)

                #To write class name on the top of the annotation
                cv2.putText(
                    img=img,
                    text=f"{classes}  {confidence}%",
                    org=(text_x, text_y), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=font_scale, 
                    color=color_scheme["label"], 
                    thickness=2
                ) 
            cv2.imwrite(image_path, img=img)
            counts["Total"] = counts["InventoryBox"]
            return image_path, counts
        except:
            logger.info("Error in draw_annotations_on_image()")

    def run_detection(self):
        """
        Main function to perform detection on user input image/images.
        """
        try:
            detection_results_path = os.path.join(os.getcwd(),"Detections")
            if not os.path.exists(detection_results_path):
                os.mkdir(path=detection_results_path)
            self.model = self.load_model(model_path=f"{os.getcwd()}{os.sep}models{os.sep}inventories_detection_model.pt")
            self.labels = self.load_labels(model_file=self.model)
            saved_file_path = self.save_image(img_array=self.input_data)
            detection_result = self.perform_inference(filepath=saved_file_path, model=self.model)
            annot_img_path, obj_counts = self.draw_annotations_on_image(image_path=saved_file_path, inference_data=detection_result)
            return annot_img_path, obj_counts
        except:
            logger.error("Error in run_detection inventories_detection()")
