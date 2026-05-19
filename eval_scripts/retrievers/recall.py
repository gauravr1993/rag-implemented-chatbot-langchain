import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

from src.config import VECTORSTORE_PATH

# Load SentenceTransformer model for semantic similarity check
st_model = SentenceTransformer("all-mpnet-base-v2")

# Load embeddings and FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)

k = 2

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": k, "score_threshold": 0.2}
)

# Load test set
with open("rag_test_queries.json", "r") as f:
    test_data = json.load(f)

# Run Retrieval Evaluation
correct = 0
total = len(test_data)
similarity_threshold = 0.6  # semantic match threshold

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
