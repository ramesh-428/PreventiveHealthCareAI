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

# -------------------- ENV --------------------
load_dotenv()

st.set_page_config(
    page_title="Preventive Healthcare AI",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# MEDIUM DARK • PREMIUM • CONSISTENT THEME
# ==========================================================
st.markdown("""
<style>
:root {
    --bg: #0b1220;
    --card: #111827;
    --surface: #0f172a;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #38bdf8;
    --accent2: #818cf8;
    --border: rgba(255,255,255,0.08);
}

/* App background */
.stApp {
    background: linear-gradient(180deg, #0b1220, #020617);
    color: var(--text);
}

/* Header */
.header {
    text-align: center;
    padding: 26px;
    border-bottom: 1px solid var(--border);
}
.header h1 {
    font-size: 2.3rem;
    font-weight: 700;
}
.header span {
    color: var(--accent);
}

.glass {
    background: rgba(17, 24, 39, 0.85);
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

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
}

/* Mode pill */
.mode-pill {
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 10px;
    display: inline-block;
}
.strict {
    background: rgba(239,68,68,0.15);
    color: #fca5a5;
}
.open {
    background: rgba(34,197,94,0.15);
    color: #86efac;
}

/* Chat bubbles */
.stChatMessage {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 16px;
}

/* Chat input (top + bottom consistency FIX) */
textarea, input, .stChatInput textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #020617;
    border-radius: 14px;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid var(--border);
}


/* ================= INPUT BOX WHITE STRIP FIX ================= */

.stChatInputContainer {
    background: #020617 !important;
    border-top: 1px solid rgba(255,255,255,0.12);
}


/* FORCE CHAT TEXT VISIBILITY */
.stChatMessage p,
.stChatMessage span,
.stChatMessage li,
.stChatMessage div {
    color: #e5e7eb !important;
    font-weight: 500;
}

/* Assistant vs User subtle distinction */
[data-testid="stChatMessage"][aria-label="assistant"] {
    background: #0f172a !important;
    border: 1px solid rgba(255,255,255,0.12);
}

[data-testid="stChatMessage"][aria-label="user"] {
    background: #020617 !important;
    border: 1px solid rgba(255,255,255,0.18);
}

/* ========== FIX 1: Chat Input Box ========== */
textarea, 
input, 
.stTextInput input,
.stTextArea textarea,
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #e5e7eb !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    padding: 12px !important;
}

/* Placeholder text in input */
textarea::placeholder,
input::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

/* Focus state for input */
textarea:focus, 
input:focus,
div[data-testid="stChatInput"] textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    outline: none !important;
}

/* ========== FIX 2: Toggle/Checkbox (Open Knowledge) ========== */
div[data-testid="stCheckbox"] {
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 12px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

div[data-testid="stCheckbox"] label {
    color: #ffffff !important;
    font-weight: 500 !important;
}

div[data-testid="stCheckbox"] label span {
    color: #ffffff !important;
}

/* ========== FIX 3: Sidebar Caption ========== */
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

.stCaption {
    color: #6b7280 !important;
}

section[data-testid="stSidebar"] .stCaption {
    color: #6b7280 !important;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div class="glass glass-title" style="text-align:center;">
    <h1>🩺 Preventive Healthcare AI Chatbot</h1>
    <p class="subtitle">
        CDAC-AI Students 2026 Project | Retrieval Augmented Generation (RAG)
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR — QUICK PROMPTS (SHIFTED HERE ✔)
# ==========================================================
with st.sidebar:
# st.sidebar.markdown("## ⚡ Quick Prompts")
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

# ==========================================================
# USER GUIDE + MODE
# ==========================================================
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

@st.cache_resource
def strict_chain():
    return get_strict_rag_chain(
        3,
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        vector_dir,
        0
    )

@st.cache_resource
def open_chain():
    return get_open_rag_chain(
        3,
        st.secrets["OPENAI_EMBEDDING_MODEL"],
        st.secrets["OPENAI_GPT_MODEL"],
        vector_dir,
        0
    )

# ==========================================================
# CHAT
# ==========================================================
st.markdown("## 💬 Chat")

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

# response generation
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Generating response..."):
        response = (
            open_chain().invoke({"input": prompt})
            if use_open else
            strict_chain().invoke({"input": prompt})
        )

        answer = response.get("answer", "No response generated")
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)