from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding import create_embeddings

from pinecone_config import pincone_store_vectors


pdf_dir = Path("./document")


def load_fin_docs():
    """
    Load all required financial docs
    """

    for pdf in pdf_dir.glob("*.pdf"):

        loader = PyPDFLoader(pdf)

        chunked_doc = create_chunks(loader.load())

        embeddings = create_embedding(chunked_doc)

        pincone_store_vectors(embeddings, chunked_doc)
        





def create_chunks(doc):
    """
    Divide the document into chunks 
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # chunked the documents
    chunkedDocs = text_splitter.split_documents(doc)

    print(f"Created {len(chunkedDocs)} chunks")

    return chunkedDocs






def create_embedding(chunked_doc):
    """
    Creates the embedding of chunked document using BAAI/bge-m3
    """
    texts = [doc.page_content for doc in chunked_doc]

    # created embeddings
    embeddings = create_embeddings(texts)

    print("Number of embeddings: ", len(embeddings))
    print("Embeddings dimensions: ", len(embeddings[0]))

    return embeddings


