import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pickle
import io

app = FastAPI()


class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: float
    max_power: float
    seats: float


@app.post("/predict_item")
def predict_item(item: Item):
    data = item.dict()
    loaded_model = pickle.load(
        open('/Users/nikitademidenko/Documents/GitHub/MLDS_ML_2022/Hometasks/HT1/Lasso_Regression_main.pkl', 'rb'))
    data_in = [[data['year'], data['km_driven'], data['mileage'], data['engine'],data['max_power'], data['seats']]]
    prediction = loaded_model.predict(data_in)
    return {
        'prediction': round(prediction[0], 2)
    }


@app.post("/predict_items")
def predict_items(csv_file: UploadFile = File(...)):
    df = pd.read_csv(csv_file.file)
    df = df.drop(columns="Unnamed: 0")
    loaded_model = pickle.load(
        open('/Users/nikitademidenko/Documents/GitHub/MLDS_ML_2022/Hometasks/HT1/Lasso_Regression_main.pkl', 'rb'))
    result = loaded_model.predict(df)

    df["price"] = result

    stream = io.StringIO()

    df.to_csv(stream,  index=False)

    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )

    response.headers["Content-Disposition"] = "attachment; filename=predicted.csv"

    return response

