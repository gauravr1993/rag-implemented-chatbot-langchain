# app.py
import streamlit as st
from src.qa_chain import create_chain  # your RAG pipeline
from utils.supabase_client import supabase  # your Supabase client
import uuid, threading
from src.pipeline import run_pipeline


# Load Retriver
@st.cache_resource
def load_retriever():
    return run_pipeline()

retriever = load_retriever()

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

qa = create_chain(retriever, session_id=st.session_state.session_id)

def save_feedback(entry):
    try:
        supabase.table("feedback").insert(entry).execute()
        print("Feedback saved to Supabase")
    except Exception as e:
        print("Error saving feedback:", e)

def handle_user_query(qa, user_input):
    smalltalk_responses = {
        "thanks": "You're welcome! Happy to help 🌱",
        "thank you": "Always here to support you 👍",
        "ok": "Anything else you'd like to know? 🤔",
        "okay": "Sure thing! What else can I assist with? 😊",
        "bye": "Goodbye! Wishing you good harvests 🌾",
        "hello": "Hello! How can I help with your crops today? 👋",
        "hi": "Hi there! What agricultural info do you need? 🌻",
        "hey": "Hey! Ask me anything about farming 🌽",
        "how are you": "I'm just a bot, but I'm here to help you with your farming questions! 🚜",
        "what's your name": "I'm Agribot, your farming assistant! 🌽",
        "help": "Sure! Ask me anything about crops, pests, or weather. 🌦️",
    }

    cleaned_input = user_input.strip().lower()

    if cleaned_input in smalltalk_responses:
        return smalltalk_responses[cleaned_input]
    else:
        response = qa.invoke({"question": user_input})
        return response


st.title("🌾 Agribot") 

# Ask Question
user_input = st.chat_input("Ask me something about crops, pests, or weather...")

if user_input:
    result = handle_user_query(qa, user_input)
    # result = qa.invoke({"question": user_input})
    if "answer" not in result:
        answer = result  # smalltalk response
        sources = []
    else:
        answer = result["answer"]
        sources = [doc.page_content[:200] for doc in result["source_documents"]]
    print("Sources:", sources)
    # Store in session history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
    })

    # Reset feedback flag for the latest answer
    st.session_state.feedback_given = False
    st.session_state.last_result = {
        "question": user_input,
        "answer": answer,
        "sources": sources
    }

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        # Render assistant with markdown for better readability
        st.chat_message("assistant").markdown(msg["content"])

if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
    if not st.session_state.feedback_given:
        cols = st.columns(2)
        if cols[0].button("👍"):
            feedback = "Helpful"
            st.session_state.feedback_given = True
        elif cols[1].button("👎"):
            feedback = "Not Helpful"
            st.session_state.feedback_given = True
        else:
            feedback = None
        
        if feedback:
            log_entry = {
                "session_id": st.session_state.session_id,
                "question": st.session_state.last_result["question"],
                "answer": st.session_state.last_result["answer"],
                "feedback": feedback,
                "source_docs": "; ".join(st.session_state.last_result["sources"])
            }
            threading.Thread(target=save_feedback, args=(log_entry,)).start()
            st.success("✅ Feedback saved")