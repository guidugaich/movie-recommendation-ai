import pymongo
from dotenv import load_dotenv
import os
from generate_embedding_hf_api import generate_embedding
from generate_embedding_local import generate_embedding_local
from generate_embedding_openai import generate_embedding_openai

load_dotenv(override=True)

# MongoDB
mongodb_uri = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(mongodb_uri)
db = client["sample_mflix"]
collection = db["embedded_movies"]

query = 'movies about war in outer space'
query_embedding = generate_embedding_openai(query)

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": "plot_embedding",
            "numCandidates": 1000,
            "limit": 4,
            "index": "vector_index"
        }
    }
])

for r in results:
    print(f'Movie: {r["title"]}\nPlot: {r["plot"]}\n\n')