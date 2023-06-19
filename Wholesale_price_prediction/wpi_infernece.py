import os
import pickle

class WPIInference():
    def __init__(self,requested_commodity) -> None:
        self.commodity = requested_commodity
        self.model = None
    def load_model(self, model_name):
        model_path = f"{os.getcwd()}{os.sep}models{os.sep}{model_name}.pkl"
        self.model = pickle.load(open(model_path,"rb"))
        return self.model
    def run_inference(self, month, year, rainfall):
        model = self.load_model(model_name=self.commodity)
        predicted_results = model.predict([[month,year,rainfall]]).item()
        return round(predicted_results,2)
    
# print(WPIInference(requested_commodity="Arhar").run_inference(month=7,year=2023,rainfall=23.43))