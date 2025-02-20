# Save embedings on database
import gc
from dotenv import load_dotenv
import os
import pymongo
from generate_embedding_hf_api import generate_embedding
from generate_embedding_local import generate_embedding_local

load_dotenv(override=True)

#mongo DB
mongodb_uri = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(mongodb_uri)
db = client["sample_mflix"]
collection = db["movies"]

docs = collection.find({
    'plot':{'$exists': True},
    'plot_embedding_768': {'$exists': False}
})

# Process in batches
batch_size = 40  # Adjust based on available memory
updates = []

for i, doc in enumerate(docs):
    embedding = generate_embedding_local(doc["plot"])
    updates.append(pymongo.UpdateOne({'_id': doc['_id']}, {'$set': {'plot_embedding_768': embedding}}))

    # Bulk write after every batch_size documents
    if len(updates) >= batch_size:
        collection.bulk_write(updates)
        updates = []  # Clear processed documents
        gc.collect()  # Force memory cleanup

if updates:
    collection.bulk_write(updates)
    gc.collect()