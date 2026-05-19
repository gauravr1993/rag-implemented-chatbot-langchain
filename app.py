from src.pipeline import run_pipeline

if __name__ == "__main__":
    qa = run_pipeline()
    while True:
        query = input("Ask a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        response = qa.invoke({"query": query})
        answer = response['answer']
        fetched_docs = response['source_documents']
        print("\nNumber of Source Documents:")
        print(len(fetched_docs))    
        print("Answer:", answer)
