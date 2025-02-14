import pymongo

from movie_recommendation import generate_embeddings

mongodb_uri = 'mongodb+srv://guidugaichdev:DaSRSFiykK4LIxqN@cluster0.qvc7v.mongodb.net/'

query = 'movies about couples that separated'

client = pymongo.MongoClient(mongodb_uri)
db = client["sample_mflix"]
collection = db["embedded_movies"]

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embeddings(query),
            "path": "plot_embedding",
            "numCandidates": 10,
            "limit": 4,
            "index": "embedded_movies_plot_index"
        }
    }
])

for r in results:
    print(f'Movie name: {r["title"]} Movie plot: {r["plot"]}')

# error: knnVector field is indexed with 1536 dimensions but queried with 384