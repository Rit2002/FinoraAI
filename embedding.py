from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cpu"
)



def create_embeddings(texts: list[str]):
    """
    Convert a list of texts into embeddings.
    """
    embedding = model.encode(
        texts,
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embedding.tolist()

