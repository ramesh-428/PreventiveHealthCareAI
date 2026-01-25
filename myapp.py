# ==========================================================
# Preventive Healthcare AI Chatbot – RAG Demo
# CDAC-AI Students 2026 Project
# ==========================================================

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from get_strict_rag_chain import get_strict_rag_chain
from get_open_rag_chain import get_open_rag_chain

# -------------------- ENV --------------------
load_dotenv()

st.set_page_config(
    page_title="Preventive Healthcare AI Chatbot",
    page_icon="🩺",
    layout="wide"
)

# -------------------- GLOBAL CSS --------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-image:
        linear-gradient(rgba(15,23,42,0.78), rgba(15,23,42,0.78)),
        url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Glass card */
.glass {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff;
}

.glass-title h1 {
    margin-bottom: 5px;
}

/* Headings */
h1, h2, h3 {
    color: #e5e7eb;
}

/* Buttons */
.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #1e293b, #020617);
    color: #e0f2fe;
    border-radius: 10px;
    font-weight: 500;
    border: 1px solid #38bdf8;
    padding: 0.45rem 0.9rem;
}
.stButton button:hover {
    background: linear-gradient(135deg, #020617, #020617);
    border-color: #67e8f9;
}

/* Chat input focus fix */
textarea:focus, input:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px #38bdf8 !important;
    border-color: #38bdf8 !important;
}

/* Chat bubbles */
.stChatMessage {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="glass glass-title" style="text-align:center;">
    <h1>🩺 Preventive Healthcare AI Chatbot</h1>
    <p class="subtitle">
        CDAC-AI Students 2026 Project | Retrieval Augmented Generation (RAG)
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------- LAYOUT --------------------
left_col, center_col, right_col = st.columns([1.3, 3.2, 1.7])

# -------------------- LEFT PANEL --------------------
with left_col:
    st.markdown("""<div class="glass"style="text-align:center;">
    <h3>💡Sample Questions </h3>
    """, unsafe_allow_html=True)

    prompts = [
        "What are the symptoms of Diabetes?",
        "How can Dengue be prevented?",
        "How to manage Asthma?",
        "Malaria vs Typhoid",
        "What is Anaemia?"
    ]

    for p in prompts:
        if st.button(p):
            st.session_state["selected_prompt"] = p

    # st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("⚠️ Educational use only. Not medical advice.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- CONFIG --------------------
current_dir = Path.cwd()
vector_dir = current_dir / os.getenv("VECTOR_DIR")
knn = int(os.getenv("KNN"))
gpt_model_creativity = int(os.getenv("OPENAI_GPT_MODEL_CREATIVITY"))

@st.cache_resource(show_spinner=False)
def load_strict_rag():
    return get_strict_rag_chain(
        knn,
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        vector_dir,
        gpt_model_creativity
    )

@st.cache_resource(show_spinner=False)
def load_open_rag():
    return get_open_rag_chain(
        knn,
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        vector_dir,
        gpt_model_creativity
    )

# -------------------- CENTER PANEL --------------------
with center_col:
    st.markdown("""
    <div class="glass" style="margin-bottom:20px;">
        <h3>🧭 User Guide</h3>
        <ul style="color:#e5e7eb; line-height:1.7;">
            <li>Ask health-related questions in simple language</li>
            <li>Use sample questions from the left panel</li>
            <li>Strict RAG gives document-based answers</li>
            <li>Open Knowledge gives general AI responses</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💬 Chat with Healthcare AI")
    user_choice = st.toggle("🌐 Use Open Knowledge", value=False)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if "selected_prompt" in st.session_state:
        prompt = st.session_state.pop("selected_prompt")
    else:
        prompt = st.chat_input("Ask about diseases, symptoms, prevention...")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Generating response..."):
            response = (
                load_open_rag().invoke({"input": prompt})
                if user_choice else
                load_strict_rag().invoke({"input": prompt})
            )

        answer = response.get("answer", "No response generated")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

# -------------------- RIGHT PANEL --------------------
with right_col:
    
    st.markdown("""<div class="glass"style="text-align:center;">
    <h3>🔁 RAG Intelligence Flow </h3>
    """, unsafe_allow_html=True)

    st.image("assets/rag.gif", use_container_width=True)

    # st.markdown(
    #     "<p style='color:#e5e7eb; margin-top:12px;'>"
    #     "1. User query received<br>"
    #     "2. Converted to embeddings<br>"
    #     "3. Vector DB retrieves context<br>"
    #     "4. LLM generates grounded response"
    #     "</p>",
    #     unsafe_allow_html=True
    # )

    
