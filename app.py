import streamlit as st
import joblib
import re
import pdfplumber
import pandas as pd
from io import BytesIO
import os, json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="AI Resume Shortlister", page_icon="📂", layout="wide")

import base64
from PIL import Image
from io import BytesIO
import streamlit as st

def image_to_base64(image_path):
    buffer = BytesIO()
    Image.open(image_path).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# your logo path here
logo_base64 = image_to_base64("logo.png")

# make Streamlit header padding smaller (so logo sits higher)
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# top-left logo (pinned)
st.markdown(
    f"""
    <div style="
        position: absolute;
        top: 38px;         /* adjust 10–20px for your screen */
        left: 10px;        /* small gap from sidebar */
        z-index: 9999;
    ">
        <img src="data:image/png;base64,{logo_base64}" width="300" style="border-radius:6px;">
    </div>
    """,
    unsafe_allow_html=True
)

# 👇 keep this after your logo code, before the title
st.markdown("""
    <style>
        h2 {
            margin-top: 100px !important;  /* push just the title, not the logo */
        }
    </style>
""", unsafe_allow_html=True)

# ===========================================
# 2️⃣ Load ML Model & Vectorizer
# ===========================================
clf = joblib.load("software_roles_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===========================================
# 3️⃣ Utility Functions
# ===========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # 🧠 Focused extraction from "Skills" and "Projects" sections
    skills_section = re.findall(r"skills?\s*[:\-–]\s*(.?)(?:\n[A-Z][a-zA-Z ]:|\Z)", text, re.IGNORECASE | re.DOTALL)
    projects_section = re.findall(r"projects?\s*[:\-–]\s*(.?)(?:\n[A-Z][a-zA-Z ]:|\Z)", text, re.IGNORECASE | re.DOTALL)
    focus_text = " ".join(skills_section + projects_section)

    # Combine full text with extracted skills/projects
    combined_text = text + " " + focus_text
    return combined_text

def extract_candidate_name(text):
    match = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", text)
    return match.group(1) if match else "Unknown"

# ===========================================
# 4️⃣ RAG Setup
# ===========================================
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    index = faiss.IndexFlatL2(384)
    st.sidebar.success("✅ Embeddings loaded successfully")
except Exception as e:
    st.sidebar.warning(f"⚠ Embeddings unavailable (falling back): {e}")
    embedder, index = None, None

kb_texts = []

def update_rag_knowledge(text, label, role_name="HR Feedback"):
    if embedder and index:
        embedding = embedder.encode([text])
        index.add(np.array(embedding, dtype="float32"))
        kb_texts.append((text, label, role_name))

def rag_filter(text, top_k=3):
    if not embedder or index.ntotal == 0:
        return "Unknown", 0.0
    query_emb = embedder.encode([text])
    D, I = index.search(np.array(query_emb, dtype="float32"), top_k)
    labels = [kb_texts[i][1] for i in I[0]]
    if labels.count("✅ Shortlisted") > labels.count("❌ Rejected"):
        return "✅ Shortlisted", 0.9
    elif labels.count("❌ Rejected") > labels.count("✅ Shortlisted"):
        return "❌ Rejected", 0.9
    return "Unknown", 0.5

# ===========================================
# 5️⃣ Persistent Feedback Storage
# ===========================================
FEEDBACK_FILE = "feedback.json"

def save_feedback(text, label, role):
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            try:
                feedback = json.load(f)
            except json.JSONDecodeError:
                feedback = []
    else:
        feedback = []
    feedback.append({"text": text, "label": label, "role": role})
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            try:
                feedback = json.load(f)
            except json.JSONDecodeError:
                feedback = []
        for fb in feedback:
            update_rag_knowledge(fb["text"], fb["label"], fb["role"])

load_feedback()

# ===========================================
# 6️⃣ Local LLM (GPT-2 Medium)
# ===========================================
try:
    llm = pipeline("text-generation", model="gpt2-medium", device=0 if torch.cuda.is_available() else -1)
    st.sidebar.success("✅ GPT-2-Medium loaded successfully.")
except Exception as e:
    st.sidebar.warning(f"⚠ LLM not loaded: {e}")
    llm = None

# ===========================================
# 7️⃣ Sidebar Controls
# ===========================================
st.sidebar.header("⚙ Settings")
threshold = st.sidebar.slider("Shortlist Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.info("Higher threshold = stricter shortlisting")
if embedder:
    st.sidebar.text(f"RAG Knowledge Size: {index.ntotal} items")

role_options = ["Data Analyst", "Web Developer", "ML Engineer", "Python Developer", "Software Engineer"]
selected_role = st.sidebar.selectbox("🎯 Select the role:", role_options)
st.sidebar.write(f"Shortlisting for *{selected_role}*")

if "last_role" not in st.session_state:
    st.session_state["last_role"] = selected_role
if st.session_state["last_role"] != selected_role:
    st.session_state["results"] = []
    st.session_state["last_role"] = selected_role
    st.info(f"🔄 Switched to {selected_role} role — starting fresh!")

results = st.session_state.get("results", [])

# ===========================================
# 8️⃣ Resume Processing with Interactive UI
# ===========================================
st.markdown("""
    <style>
    .main-title { text-align:center; font-size:2.2em; color:#2E86C1; font-weight:700; }
    .candidate-card { padding:15px; border-radius:15px; margin-bottom:20px; background-color:#f9f9fb; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
    .shortlisted { border-left: 5px solid #2ECC71; }
    .rejected { border-left: 5px solid #E74C3C; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='main-title'>📂 AI Resume Shortlister</h2>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("📄 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

progress_placeholder = st.empty()

if uploaded_files:
    results.clear()
    progress_bar = st.progress(0)
    for idx, file in enumerate(uploaded_files, start=1):
        progress_bar.progress(idx / len(uploaded_files))
        progress_placeholder.text(f"Processing {file.name}...")

        text = extract_text_from_pdf(file)
        cleaned = clean_text(text)
        name = extract_candidate_name(text)

        # --- ML + RAG Inference ---
        features = vectorizer.transform([cleaned])
        prob = max(clf.predict_proba(features)[0])
        rag_label, _ = rag_filter(cleaned)

        role_keywords = {
            "Data Analyst": ["excel", "tableau", "sql", "analytics", "power bi", "statistics", "data analysis"],
            "Web Developer": ["html", "css", "javascript", "react", "frontend", "backend", "node", "api"],
            "ML Engineer": ["machine learning", "tensorflow", "pytorch", "neural network", "deep learning"],
            "Python Developer": ["python", "django", "flask", "automation", "pandas"],
            "Software Engineer": ["java", "c++", "git", "software design", "oop", "spring"]
        }
        matched_keywords = [kw for kw in role_keywords.get(selected_role, []) if kw in cleaned]

        # --- Scoring logic ---
        base_prob = prob + 0.1 * len(matched_keywords)
        if selected_role.lower() in cleaned:
            base_prob += 0.05
        if rag_label == "✅ Shortlisted":
            base_prob += 0.1
        base_prob = min(base_prob, 1.0)

        final_score = round(base_prob, 2)
        decision = "✅ Shortlisted" if base_prob >= threshold else "❌ Rejected"

        # --- GPT-2 Reason ---
        reason_base = f"The resume shows skills like {', '.join(matched_keywords[:5]) or 'limited relevant experience'}."
        llm_reason = reason_base
        if llm:
            try:
                prompt = (
                    f"The Candidate is {decision.lower()} "
                    f"for a {selected_role} position based on these skills: {', '.join(matched_keywords[:5])}."
                )
                output = llm(prompt, max_new_tokens=60, temperature=0.8)
                response = output[0]["generated_text"].strip()
                llm_reason = response.split(".")[0].strip().capitalize() + "."
            except Exception:
                llm_reason = reason_base

        results.append({
            "ID": idx,
            "Resume": file.name,
            "Name": name,
            "Target Role": selected_role,
            "Decision": decision.replace('✅ ', '').replace('❌ ', ''),
            "Confidence": round(base_prob * 100, 1),
            "Reason": llm_reason,
            "HR Feedback": ""
        })

    st.session_state["results"] = results
    progress_placeholder.empty()
    progress_bar.empty()
    st.success("✅ All resumes processed!")

# ===========================================
# 9️⃣ Final Polished Candidate Results UI
# ===========================================
if results:
    st.markdown(
        """
        <style>
        .results-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .candidate-card {
            width: 80%;
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 25px;
            background-color: #f9f9fb;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            text-align: center;
            transition: 0.3s ease;
        }
        .candidate-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        }
        .shortlisted {
            border-top: 6px solid #2ecc71;
            background: linear-gradient(180deg, #eafaf1 0%, #f9f9fb 100%);
        }
        .rejected {
            border-top: 6px solid #e74c3c;
            background: linear-gradient(180deg, #fdecea 0%, #f9f9fb 100%);
        }
        .candidate-name {
            font-size: 1.6em;
            font-weight: 700;
            color: #2c3e50;
        }
        .target-role {
            font-size: 1.2em;
            font-weight: 600;
            color: #2E86C1;
        }
        .decision {
            font-size: 1.3em;
            font-weight: 700;
            margin-top: 10px;
        }
        .decision-shortlisted {
            color: #27ae60;
        }
        .decision-rejected {
            color: #c0392b;
        }
        .reason {
            font-size: 1.05em;
            color: #444;
            margin-top: 15px;
            background: #f0f3f5;
            border-radius: 10px;
            padding: 12px;
        }
        .feedback-buttons {
            margin-top: 12px;  /* ≈0.5 inch spacing from reason box */
            display: flex;
            justify-content: center;
            gap: 20px;
        }
        .feedback-btn {
            border: none;
            color: white !important;
            font-weight: 600;
            border-radius: 10px;
            padding: 10px 22px;
            cursor: pointer;
            transition: 0.2s ease;
            font-size: 1em;
        }
        .feedback-btn:hover {
            opacity: 0.9;
            transform: scale(1.03);
        }
        .confirm-btn {
            background-color: #27ae60;
        }
        .reject-btn {
            background-color: #e74c3c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3 style='text-align:center;'>📋 Candidate Decisions</h3>", unsafe_allow_html=True)
    st.markdown("<div class='results-container'>", unsafe_allow_html=True)

    for i, row in enumerate(results):
        card_class = "shortlisted" if row["Decision"] == "Shortlisted" else "rejected"
        decision_class = (
            "decision-shortlisted" if row["Decision"] == "Shortlisted" else "decision-rejected"
        )

        st.markdown(f"<div class='candidate-card {card_class}'>", unsafe_allow_html=True)

        # 🧍 Candidate header
        st.markdown(f"<div class='candidate-name'>🧍 {row['Name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='target-role'>🎯 Target Role: {row['Target Role']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='decision {decision_class}'>📊 Decision: {row['Decision']} ({row['Confidence']}%)</div>",
            unsafe_allow_html=True,
        )

        # 💬 Direct reasoning
        st.markdown(f"<div class='reason'>💬 <b>Reason:</b> {row['Reason']}</div>", unsafe_allow_html=True)

               # 🧠 Feedback buttons (styled inline with background colors)
        st.markdown("""
        <div style='display: flex; justify-content: left; gap: 20px; margin-top: 15px;'>
            <style>
                .btn-confirm {
                    background-color: #27ae60;
                    color: white !important;
                    font-weight: 600;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 28px;
                    font-size: 1em;
                    cursor: pointer;
                    transition: 0.2s ease;
                }
                .btn-confirm:hover {
                    background-color: #229954;
                    transform: scale(1.03);
                }
                .btn-reject {
                    background-color: #e74c3c;
                    color: white !important;
                    font-weight: 600;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 28px;
                    font-size: 1em;
                    cursor: pointer;
                    transition: 0.2s ease;
                }
                .btn-reject:hover {
                    background-color: #c0392b;
                    transform: scale(1.03);
                }
            </style>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1.5, 1, 1.5])
        with col1:
            pass  # empty for centering
        with col2:
            confirm_btn = st.button(f"✅ Confirm {row['Name']}", key=f"confirm_{i}")
        with col3:
            reject_btn = st.button(f"❌ Reject {row['Name']}", key=f"reject_{i}")

        # Handle button logic
        if confirm_btn:
            save_feedback(row["Reason"], "✅ Shortlisted", selected_role)
            update_rag_knowledge(row["Reason"], "✅ Shortlisted", selected_role)
            st.success(f"Confirmed {row['Name']} as correct.")
        if reject_btn:
            save_feedback(row["Reason"], "❌ Rejected", selected_role)
            update_rag_knowledge(row["Reason"], "❌ Rejected", selected_role)
            st.error(f"Rejected {row['Name']} as incorrect.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Excel Export
    buffer = BytesIO()
    df = pd.DataFrame(results)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")

    st.sidebar.download_button(
        label=f"📥 Download {selected_role}_Results.xlsx",
        data=buffer,
        file_name=f"{selected_role}_shortlist_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
