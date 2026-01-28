# ==========================================================
# Preventive Healthcare AI Chatbot
# CDAC-AI Students 2026 | RAG Medical Assistant
# ==========================================================

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from get_strict_rag_chain import get_strict_rag_chain
from get_open_rag_chain import get_open_rag_chain

#Page Config
st.set_page_config(
    page_title="Preventive Healthcare AI",
    page_icon="🩺",
    layout="wide"
)
#Load CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Inject the CSS file
local_css("styles.css")

#Page Title
st.markdown("""
<div class="glass glass-title" style="text-align:center;">
    <h1>🩺 Preventive Healthcare AI Chatbot</h1>
    <p class="subtitle">
        CDAC-AI Students 2026 Project | Retrieval Augmented Generation (RAG)
    </p>
</div>
""", unsafe_allow_html=True)

# Left Side Bar Quick Prompts
with st.sidebar:
    st.markdown("""
        <div class="glass" style="text-align:center; padding: 12px; margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 1.3rem;">⚡ Quick Prompts</h3>
        </div>
        """, unsafe_allow_html=True)
    quick_prompts = [
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

    for qp in quick_prompts:
        if st.button(qp, use_container_width=True):
            st.session_state["pending_prompt"] = qp
            
    st.markdown("---")
    st.caption("Click to instantly test the RAG system")

# User Guide (Center)
left, right = st.columns([3,1])

with left:
    st.markdown("""
    <div class="card">
        <h3>📘 User Guide</h3>
        <ul>
            <li>Ask preventive healthcare questions via chat</li>
            <li>Use sidebar prompts for instant demos</li>
            <li><b>Strict Mode</b> → answers only from medical PDFs</li>
            <li><b>Open Mode</b> → wider AI medical reasoning</li>
            <li>Compare both modes for evaluation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""<div class="card" style="text-align:center; padding: 11px; margin-bottom: 11px;">
    <h3 style="margin: 0; font-size: 1.3rem;"> ⚙️ Mode Control</h3>
    """, unsafe_allow_html=True)
    # st.markdown("### ⚙️ Mode Control")

    use_open = st.toggle("Open Knowledge", value=False)

    mode_class = "open" if use_open else "strict"
    mode_text = "OPEN MODE" if use_open else "STRICT MODE"

    st.markdown(
        f'<div class="mode-pill {mode_class}">{mode_text}</div>',
        unsafe_allow_html=True
    )

    show_rag = st.button("🧠 View RAG Flow")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# RAG FLOW (NO CHANGE — WOW PART 🔥)
# ==========================================================
if show_rag:
    st.markdown("""
    <div class="card">
        <h3>🔁 RAG Intelligence Flow</h3>
        <p style="color:#9ca3af;">
        How the query is retrieved, grounded and answered.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image("assets/strict_rag.png", caption="Strict RAG Flow", use_container_width=True)
    with c2:
        st.image("assets/open_rag.png", caption="Open RAG Flow", use_container_width=True)

# ==========================================================
# RAG CHAINS
# ==========================================================
current_dir = Path.cwd()
vector_dir = current_dir / os.getenv("VECTOR_DIR", "chroma_db")

@st.cache_resource(show_spinner=False)
def strict_chain():
    return get_strict_rag_chain(
        st.secrets["KNN"],
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        vector_dir,
        st.secrets["OPENAI_GPT_MODEL_CREATIVITY"],
        st.secrets["MAX_OUTPUT_CHARS"]
    )

@st.cache_resource(show_spinner=False)
def open_chain():
    return get_open_rag_chain(
        st.secrets["KNN"],
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        vector_dir,
        st.secrets["OPENAI_GPT_MODEL_CREATIVITY"],
        st.secrets["MAX_OUTPUT_CHARS"]
    )

# ==========================================================
# CHAT
# ==========================================================
st.markdown("## Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# chat input ALWAYS visible
user_input = st.chat_input("Ask about diseases, symptoms, prevention...")

# unified prompt selector
prompt = None
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")
elif user_input:
    prompt = user_input

#Get prior chat message and response
prior_chat=st.session_state["messages"][-2:]

# response generation
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Generating response..."):
        response = (
            open_chain().invoke({"input": prompt, "chat_history":prior_chat})
            if use_open else
            strict_chain().invoke({"input": prompt,"chat_history":prior_chat})
        )

        answer = response.get("answer", "No response generated")
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)