from langchain_core.prompts import ChatPromptTemplate

from rag_chain import make_rag_chain
from langchain_groq import ChatGroq

MODEL = "llama3-8b-8192"


def get_model():
    llm = ChatGroq(model=MODEL, temperature=0.1, streaming=True,
                   api_key="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")
    return llm


def create_full_chain(retriever):
    model = get_model()

    #TODO: Need to find a way to implement prompt

    # system_prompt = """You are a helpful AI assistant for history explorers who wants to learn more about Greek
    # Mythology. Use the following context and the users' chat history to help the user: If you don't know the
    # answer, just say that you don't know.
    #
    # Context: {context}
    #
    # Question: """
    #
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         ("human", "{question}"),
    #     ]
    # )

    rag_chain = make_rag_chain(model, retriever)
    return rag_chain


def ask_question(chain, query, chat_history):
    response = chain(
        {"question": query, "chat_history": chat_history}
    )
    return response
