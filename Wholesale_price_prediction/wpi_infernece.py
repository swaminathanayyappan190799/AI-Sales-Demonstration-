import os
import pickle
from log_settings import logger

class WPIInference():
    def __init__(self,requested_commodity) -> None:
        self.commodity = requested_commodity
        self.model = None
    def load_model(self, model_name):
        model_path = f"{os.getcwd()}{os.sep}models{os.sep}{model_name}.pkl"
        try:
            self.model = pickle.load(open(model_path,"rb"))
            logger.info(f"Model loaded for {model_name}")
        except:
            logger.error("Model not found")
        return self.model
    def run_inference(self, month, year, rainfall):
        try:
            model = self.load_model(model_name=self.commodity)
            predicted_results = model.predict([[month,year,rainfall]]).item()
            logger.info(f"The WPI is {predicted_results}")
            return round(predicted_results,2)
        except:
            logger.error("Error in run_inference()")
            return None
