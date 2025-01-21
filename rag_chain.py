from langchain.chains import ConversationalRetrievalChain


def make_rag_chain(model, retriever):

    rag_chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, verbose=True)

    return rag_chain
