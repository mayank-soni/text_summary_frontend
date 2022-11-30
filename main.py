from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {'Hello': 'World'}

#http://127.0.0.1:8000/predict?category=sports&language=en&q=worldcup
@app.get('/predict')
def predict(keywords: str,              #e.g. world cup
            category: str = 'sports'
):
    url = 'https://newsdata.io/api/1/news'
    params = {"apikey": os.environ['newsData_API_key'], "category":f"{category}", "language":"en", "q":f"{keywords}"}
    response = requests.get(url, params=params)
    return response.json()
