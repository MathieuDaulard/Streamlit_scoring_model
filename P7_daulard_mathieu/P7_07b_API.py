import joblib
from lightgbm.sklearn import LGBMClassifier
import pandas as pd
import numpy as np
import json

class Scoring_model():
    data = pd.read_csv("DATA_SCALED.csv")
    data.set_index("SK_ID_CURR", drop= True, inplace =True)
    
    def __init__(self, model_path:str):
        self.model = self.get_model(model_path)
        self.solvabilite = {
            0:'Solvable',
            1:'Non Solvable'
        }
    
    def get_id(self):
        return self.data.index

    def get_model(self, model_path:str) -> LGBMClassifier:
        '''Open the joblib file which store the model.
        Arguments: 
            model_path: Path model with joblib extension
        
        Returns:
            model: Model object
        '''

        with open(model_path,"rb") as f:
            model = joblib.load(f)
        
        return model

    def make_prediction(self, id_client):
        data = self.data
        data = data.loc[id_client].values
        if data.ndim == 1:
            data = data.reshape(1,-1)
        pred = self.model.predict_proba(data)
        pred = pd.Series(np.where(pred[:,1]>0.27,self.solvabilite[1], self.solvabilite[0]))
        return pred