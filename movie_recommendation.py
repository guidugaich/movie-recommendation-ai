# Creating embedings for movie data using feature-extraction pipeline API from huggingface
from dotenv import load_dotenv
import os
import pymongo
import requests
import pandas as pd

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
mongodb_uri = os.getenv("MONGODB_URI")
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{embedding_model_id}"
headers = {"Authorization": f"Bearer {huggingface_token}"}

client = pymongo.MongoClient(mongodb_uri)
db = client["sample_mflix"]
collection = db["movies"]

texts = []
for movie in collection.find().limit(5):
    texts.append(movie["plot"])

def generate_embeddings(text: str) -> list[float]:
    response = requests.post(api_url, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        raise Exception("API request failed with status code: ", response.status_code)

    return response.json()

print(pd.DataFrame(generate_embeddings(texts)))

# now we should create Embeddings - vectors of numbers that represent the movie data