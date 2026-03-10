from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pickle
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = pickle.load(open("model.pkl","rb"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            pclass: int = Form(...),
            sex: int = Form(...),
            age: float = Form(...),
            sibsp: int = Form(...),
            parch: int = Form(...),
            fare: float = Form(...)):

    data = np.array([[pclass,sex,age,sibsp,parch,fare]])
    prediction = model.predict(data)

    result = "Survived" if prediction[0]==1 else "Did Not Survive"

    return templates.TemplateResponse("result.html",
            {"request": request, "prediction": result})