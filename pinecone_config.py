import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


pinecone = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# print(pinecone.list_indexes())

pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

def pincone_store_vectors(embeddings, chunkedDocs):

    BATCH_SIZE = 100

    for start in range(0, len(chunkedDocs), BATCH_SIZE):

        batch_docs = chunkedDocs[start : start + BATCH_SIZE]
        batch_embeddings = embeddings[start : start + BATCH_SIZE]

        vectors = []

        for i, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):

            vectors.append({
                "id" : f"chunk-{start + i}",
                "values" : embedding,
                "metadata": {
                    "text" : doc.page_content,
                    "source" : doc.metadata.get("source", ""),
                    "page" : doc.metadata.get("page", 0)
                }
            })

        pinecone_index.upsert(vectors=vectors)

        print(f"Uploaded {start + len(batch_docs)} / {len(chunkedDocs)}")



