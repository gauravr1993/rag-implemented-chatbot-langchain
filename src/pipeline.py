from src.document_loader import load_documents, preprocess_documents
from src.vector_store import update_vector_store
from src.qa_chain import get_retriever

def run_pipeline():
    docs = load_documents()
    chunks = preprocess_documents(docs)
    update_vector_store(chunks)
    retriever = get_retriever(chunks)
    return retriever