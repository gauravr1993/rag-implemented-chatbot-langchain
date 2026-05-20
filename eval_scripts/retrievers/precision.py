import sys
import json
from pathlib import Path

# Add project root to path so imports work correctly
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

# Load SentenceTransformer model for semantic similarity check
st_model = SentenceTransformer("all-mpnet-base-v2")

# Load embeddings and FAISS vector store with absolute paths
vectorstore_path = PROJECT_ROOT / "vectorstore" / "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local(str(vectorstore_path), embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.2}
)

def is_relevant(ground_truth, doc, threshold=0.7):
    score = util.cos_sim(st_model.encode(ground_truth), st_model.encode(doc)).item()
    return score >= threshold

# Load test set with absolute path
test_queries_path = PROJECT_ROOT / "eval_scripts" / "retrievers" / "rag_test_queries.json"
with open(test_queries_path, "r") as f:
    test_data = json.load(f)


def compute_precision_at_k(retriever, test_queries, k=2):
    total_precision = 0
    detailed_logs = []

    for item in test_queries:
        query = item["query"]
        ground_truth = item["ground_truth"].lower()
        
        # Retrieve top-k documents
        docs = retriever.get_relevant_documents(query)
        retrieved_contents = [doc.page_content.lower() for doc in docs[:k]]
        
        # Count how many are relevant
        relevant_count = sum([1 for doc in retrieved_contents if is_relevant(ground_truth, doc, threshold=0.7)])
        precision = relevant_count / k

        detailed_logs.append({
            "query": query,
            "precision@k": precision,
            "retrieved_count": len(docs),
            "relevant_found": relevant_count,
            "matched_texts": [doc for doc in retrieved_contents if is_relevant(ground_truth, doc, threshold=0.7)]
        })

        total_precision += precision

    avg_precision = total_precision / len(test_queries)
    return avg_precision, detailed_logs


precision, logs = compute_precision_at_k(retriever, test_data, k=2)

print(f"Detailed Retrieval Precision Logs: {json.dumps(logs, indent=2)}")
print(f"📊 Precision@3: {precision:.2f}")