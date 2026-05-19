import os
from dotenv import load_dotenv
load_dotenv()
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Build this dataset by running your chatbot on test questions
eval_data = {
    "question": ["How is the air quality changing in Dhanbad?"],      # your test questions
    "answer": ["The air quality in Dhanbad has been deteriorating due to industrial emissions."],        # chatbot responses
    "contexts": [["Information about air quality in Dhanbad", "Data on industrial emissions"]],      # retrieved chunks (list of lists)
    "ground_truth": ["The air quality in Dhanbad has been deteriorating due to industrial emissions."]   # manually written expected answers
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision], llm=llm, embeddings=embedding_model)
print(results)