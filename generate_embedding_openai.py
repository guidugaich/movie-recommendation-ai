from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

def generate_embedding_openai(text: str) -> list[float]:
    response = client.embeddings.create(
      model="text-embedding-ada-002",
      input=text,
      encoding_format="float"
    )

    return response.data[0].embedding