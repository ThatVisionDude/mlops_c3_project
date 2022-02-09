from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import json
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system(f"dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

from ml.data import process_data
from ml.model import load_model, inference
model_name = "model"
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI()

class ModelInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int=Field(alias='education-num')
    marital_status: str=Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int=Field(alias='capital-gain')
    capital_loss: int=Field(alias='capital-loss')
    hours_per_week: int=Field(alias='hours-per-week')
    native_country: str=Field(alias='native-country')
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

@app.get("/")
async def welcome_user():
    return {"welcome_message": "Welcome, Dear User"}

@app.post("/inference/")
async def doInference(inp: ModelInput):
    dico = json.loads(inp.json())
    X=pd.DataFrame(columns = dico.keys())
    X = X.append(dico, ignore_index=True)
    X.columns = [k.replace("_", "-") for k in dico.keys()]
    model, enc, lb = load_model(model_name)
    X_inp, _,_,_= process_data(
    X, training=False, 
    categorical_features=cat_features, 
    encoder = enc, lb = lb)
    inf_res = inference(model, X_inp)
    label = lb.inverse_transform(inf_res)
    return {"inference_result": label[0]}
