import os

# Paths
DATA_PATH = os.path.join("data/dicra")
VECTORSTORE_PATH = os.path.join("vectorstore", "faiss_index")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Chunking
CHUNK_SIZE = 384
CHUNK_OVERLAP = 50