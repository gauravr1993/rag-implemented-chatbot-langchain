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

k = 2

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": k, "score_threshold": 0.2}
)

# Load test set with absolute path
test_queries_path = PROJECT_ROOT / "eval_scripts" / "retrievers" / "rag_test_queries.json"
with open(test_queries_path, "r") as f:
    test_data = json.load(f)

# Run Retrieval Evaluation
correct = 0
total = len(test_data)
similarity_threshold = 0.7  # semantic match threshold

for item in test_data:
    query = item["query"]
    ground_truth = item["ground_truth"]
    # print("Query: ", query)
    # print("Ground Truth: ", ground_truth)

    # Retrieve top-k docs
    docs = retriever.get_relevant_documents(query)
    found = False

    for doc in docs:
        content = doc.page_content
        score = util.cos_sim(st_model.encode(content), st_model.encode(ground_truth))[0][0].item()
        # print("Content: ", content)
        print("Score: ", score)

        if score >= similarity_threshold:
            found = True
            break

    if found:
        correct += 1

    print(f"🔍 Query: {query[:50]}... | Match Found: {found}")

recall_at_k = correct / total
print(f"\n📊 Retrieval Recall@{k}: {recall_at_k:.2f}")
