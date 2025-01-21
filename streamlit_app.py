import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

from full_chain import create_full_chain, ask_question

from langchain.vectorstores import FAISS

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    if "chat_history" not in st.session_state.keys():
        st.session_state.chat_history = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt, st.session_state.chat_history[-5:])
                st.session_state.chat_history.append((prompt, response['answer']))
                st.markdown(response['answer'])
        message = {"role": "assistant", "content": response['answer']}
        st.session_state.messages.append(message)


@st.cache_resource
def get_retriever():
    # docs = load_txt_files()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("/Users/chaitanyamalik/greek_vs", embeddings,
                          allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"k": 3, 'score_threshold': 0.2})
    return retriever


def get_chain():
    retriever = get_retriever()
    chain = create_full_chain(retriever)
    return chain


def run():
    ready = True
    api_key = "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"

    if ready:
        chain = get_chain()
        st.subheader("Ask me questions about Greek Mythology")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()


run()
