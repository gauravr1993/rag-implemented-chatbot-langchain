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
    search_kwargs={"k": 5, "score_threshold": 0.3}  # use larger k for MRR evaluation
)

# Load test set with absolute path
test_queries_path = PROJECT_ROOT / "eval_scripts" / "retrievers" / "rag_test_queries.json"
with open(test_queries_path, "r") as f:
    test_data = json.load(f)

# Run Retrieval Evaluation for MRR
reciprocal_ranks = []
similarity_threshold = 0.7  # semantic match threshold

for item in test_data:
    query = item["query"]
    ground_truth = item["ground_truth"]

    docs = retriever.get_relevant_documents(query)
    rank = None  # track at which rank ground truth is found

    for idx, doc in enumerate(docs, start=1):
        content = doc.page_content
        score = util.cos_sim(st_model.encode(content), st_model.encode(ground_truth))[0][0].item()
        print(f"Doc Rank {idx} | Score: {score:.4f}")

        if score >= similarity_threshold:
            rank = idx
            break

    if rank:
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)

    print(f"🔍 Query: {query[:50]}... | Reciprocal Rank: {1/rank if rank else 0:.2f}")

# Compute Mean Reciprocal Rank (MRR)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
print(f"\n📊 Mean Reciprocal Rank (MRR): {mrr:.2f}")
