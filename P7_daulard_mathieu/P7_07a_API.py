from sqlite3 import DatabaseError

from h11 import Data
from scoring_code import Scoring_model
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

#create the application
app = FastAPI(
    title = "Solvabilite Classifier API",
    version = 1.0,
    description = "Simple API to make predict of client solvancy."
)

#creating the classifier
classifier = Scoring_model("LightGBMModel.joblib")


class Item(BaseModel):
    ids: list = []

@app.post("/")
async def get_prediction(id_client:Item):
    id_client = id_client.ids
    for i in id_client :
        if i not in classifier.get_id():
            raise HTTPException(status_code = 446, detail = str(i) + " non trouvé")
    result_classification = classifier.make_prediction(id_client)
    return result_classification