# creating embeddings for movie plots using SentenceTransformer

import pymongo
from sentence_transformers import SentenceTransformer
import pandas as pd

mongodb_uri = 'mongodb+srv://guidugaichdev:DaSRSFiykK4LIxqN@cluster0.qvc7v.mongodb.net/'
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
