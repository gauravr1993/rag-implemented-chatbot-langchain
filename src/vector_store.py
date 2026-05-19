import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import VECTORSTORE_PATH, EMBEDDING_MODEL
from utils.documentIDUtils import filter_new_docs, load_doc_ids, save_doc_ids

def create_vector_store(docs):
    """Create a new FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(VECTORSTORE_PATH)
    return vectordb

def load_vector_store():
    """Load existing FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def update_vector_store(new_docs):
    """Add new documents to existing FAISS store."""
    vectordb = load_vector_store()
    print(f"Loaded existing vector store: {vectordb is not None}")
    if vectordb:
        existing_ids = load_doc_ids(path=f"{VECTORSTORE_PATH}/doc_ids.json")
        print(f"Existing document IDs: {len(existing_ids)}")
        new_docs = filter_new_docs(new_docs, existing_ids)
        print(f"Adding {len(new_docs)} new documents to vector store.")
        if new_docs:
            vectordb.add_documents(new_docs)
            vectordb.save_local(VECTORSTORE_PATH)
            # update IDs
            new_ids = {d.metadata["doc_id"] for d in new_docs}
            all_ids = existing_ids.union(new_ids)
            save_doc_ids(all_ids, path=f"{VECTORSTORE_PATH}/doc_ids.json")
    else:
        vectordb = create_vector_store(new_docs)
    return vectordb
