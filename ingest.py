from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



pdf_dir = Path("./document")


def load_fin_docs():
    """Load all required financial docs"""

    docs: list = []

    for pdf in pdf_dir.glob("*.pdf"):

        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} pages")

    return docs


docs = load_fin_docs

# "\n\n" : para break
# "\n" : line break, when para becomes too large
# "." : sentence boudary
# " " : word boundary
# "" : character boundary
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunkedDocs = text_splitter.split_documents(docs)