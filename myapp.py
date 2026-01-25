# ==========================================================
# Preventive Healthcare AI Chatbot – RAG Demo
# CDAC-AI Students 2026 Project
# ==========================================================

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from process_pdf import process_pdf
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
.stApp {
    background-image:
        linear-gradient(rgba(15,23,42,0.75), rgba(15,23,42,0.75)),
        url("https://images.unsplash.com/photo-1580281657521-6dcb7caa2d39");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.glass {
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff;
}

h1, h2, h3 {
    color: #e5e7eb;
}

.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e0f2fe;
    border-radius: 10px;
    border: 1px solid #38bdf8;
    
}
.stButton button:hover {
    border-color: #67e8f9;
}

.stChatMessage {
    background: rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="glass" style="text-align:center;">
    <h1>🩺 Preventive Healthcare AI Chatbot</h1>
    <p>CDAC-AI Students 2026 | RAG Based Medical Assistant</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# LEFT SIDEBAR — SAMPLE QUESTIONS
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div class="glass" style="text-align:center; padding: 12px; margin-bottom: 12px;">
        <h3 style="margin: 0; font-size: 1.3rem;">💡 Sample Questions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    

    questions = [
        "What are the symptoms of Diabetes?",
        "How can Dengue be prevented?",
        "How to manage Asthma?",
        "Malaria vs Typhoid",
        "What is Anaemia?",
        "What are early signs of Hypertension?",
        "How to prevent Heart Disease?",
        "What causes Tuberculosis?",
        "Symptoms of COVID-19",
        "How to boost immunity naturally?"
    ]

    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state["selected_prompt"] = q

    st.markdown("---")
    st.caption("⚠️ Educational use only")

# ==========================================================
# CONFIG
# ==========================================================

@st.cache_resource(show_spinner="🔄 Initializing medical knowledge base...")
def init_vector_db():
    if not vector_dir.exists():
        process_pdf("data")   # ← yahi tumhara PDF folder

init_vector_db()

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

# ==========================================================
# MAIN CHAT AREA
# ==========================================================
st.markdown("""
<div >
    <h3 style="margin: 0; font-size: 1.3rem;">🧭 User Guide</h3>
    <ul>
        <li><strong>Ask Questions:</strong> Type any health-related query in the chat below</li>
        <li><strong>Quick Access:</strong> Click sample questions from the sidebar for instant responses</li>
        <li><strong>Strict RAG Mode:</strong> Get answers directly from verified medical documents</li>
        <li><strong>Open Knowledge Mode:</strong> Access broader AI knowledge with general health information</li>
        <li><strong>Explore More:</strong> Switch between modes using the toggle to compare responses</li>
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
     # 🔽 FORCE SCROLL TO BOTTOM AFTER NEW MESSAGE
    st.rerun()
# ==========================================================
# RIGHT PANEL — SIDEBAR-LIKE (EXPANDER)
# ==========================================================
with st.expander("🔁 RAG Intelligence Flow"):
    st.markdown("""
    <div class="glass" style="text-align:center;">
        <h3>How RAG Works</h3>
    </div>
    """, unsafe_allow_html=True)

    st.image("assets/open_rag.png", use_container_width=True)
    st.image("assets/strict_rag.png", use_container_width=True)
