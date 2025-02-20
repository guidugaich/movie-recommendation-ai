# creating embeddings for movie plots using lib SentenceTransformer
from sentence_transformers import SentenceTransformer

embedding_model_id = "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"

def generate_embedding_local(text: str) -> list[float]:
    model = SentenceTransformer(
        embedding_model_id,
        trust_remote_code=True
    )
    embeddings = model.encode(
        text,
        precision="float32",
        batch_size=64,
        show_progress_bar=True
    )
    return [float(x) for x in embeddings]
