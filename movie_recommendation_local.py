# creating embeddings for movie plots using SentenceTransformer
from dotenv import load_dotenv
import os
import pymongo
from sentence_transformers import SentenceTransformer
import pandas as pd

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"

client = pymongo.MongoClient(mongodb_uri)
db = client["sample_mflix"]
collection = db["movies"]

texts = []
for movie in collection.find().limit(5):
    texts.append(movie["plot"])

def generate_embeddings(text: str) -> list[float]:
    model = SentenceTransformer(embedding_model_id)
    embeddings = model.encode(text)
    return embeddings
    
embeddings = generate_embeddings(texts)
print(pd.DataFrame(embeddings))
