from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents():
    """Load PDF and TXT documents from data directory."""
    loaders = [
        DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader),
    ]
    docs = []
    print(loaders)
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def preprocess_documents(docs):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)