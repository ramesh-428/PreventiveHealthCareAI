#This is a custom chatbot application to demo RAG architecture using 
#Custom Docs + Open AI Model.
import streamlit as st
import os
from get_rag_chain import get_rag_chain
from pathlib import Path

# --- Page setup ---
st.set_page_config(page_title="Preventive HealthCare Chat Demo", page_icon="💬")

st.title("💬 CDAC AI BATCH PROJECT DEMO")
st.write("A simple Streamlit app to demo RAG. Answers queries related to common occuring diseases in India"
" Anaemia,Asthma,Covid-19,Dengue,Diabetes,HyperTension,Malaria,Tuberculosis and Typhoid.")

current_dir = Path.cwd() #Get Current Working Directory
vector_dir = current_dir / os.getenv("VECTOR_DIR") # Read the Vector DB Base directory path 
openai_embed_model=os.getenv("OPENAI_EMBEDDING_MODEL") #Model for Word Embeddings
openai_gpt_model=os.getenv("OPENAI_GPT_MODEL") #Chat Model to refine responses
knn=int(os.getenv("KNN")) #Neareset Neighbours for similarity search

@st.cache_resource(show_spinner=False)  # Add the caching decorator
def load_rag():
    return get_rag_chain(
        knn,
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        Path.cwd() / st.secrets["VECTOR_DIR"]
    )
# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display chat history ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    rag_response = load_rag().invoke({"input": prompt})

    # Add assistant response
    st.session_state["messages"].append({"role": "assistant", "content": rag_response['answer']})
    st.chat_message("assistant").write(rag_response['answer'])