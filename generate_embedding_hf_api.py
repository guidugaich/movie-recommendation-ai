# Creating embedings for movie data using feature-extraction pipeline API from huggingface
from dotenv import load_dotenv
import os
import requests

load_dotenv(override=True)

#Hugging face
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{embedding_model_id}"
headers = {"Authorization": f"Bearer {huggingface_token}"}

def generate_embedding(text: str) -> list[float]:
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    
    return response.json() 