import hashlib
import os
import json

def generate_doc_id(doc):
    """Generate a hash ID for a document's content."""
    content = doc.page_content.encode("utf-8")
    return hashlib.md5(content).hexdigest()

def filter_new_docs(docs, existing_ids):
    """Filter out documents that already exist in the vector store."""
    new_docs = []
    for doc in docs:
        doc_id = generate_doc_id(doc)
        if doc_id not in existing_ids:
            doc.metadata["doc_id"] = doc_id
            new_docs.append(doc)
    return new_docs

def save_doc_ids(existing_ids, path="vectorstore/doc_ids.json"):
    """Persist document IDs to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(existing_ids), f)

def load_doc_ids(path="vectorstore/doc_ids.json"):
    """Load document IDs from JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()
