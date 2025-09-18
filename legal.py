# legal.py

# Install dependencies first if not done:
# pip install streamlit sentence-transformers scikit-learn numpy requests pypdf

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import requests
from pypdf import PdfReader

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
st.title("📄 Legal Document Risk Analyzer")
st.write("Upload a legal document (.txt or .pdf) to detect high-risk clauses, see suggestions, summarize key points, and fetch related judgements from Indian Kanoon.")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "pdf"])

# -----------------------------
# Risk Solution Mapping
# -----------------------------
risk_solutions = {
    "payment": "Specify clear payment terms and criteria for approval to avoid ambiguity.",
    "liability": "Balance liability clauses; include exceptions for gross negligence.",
    "termination": "Clearly define termination conditions to avoid disputes.",
    "non-compete": "Ensure duration and scope are reasonable and enforceable.",
    "governing law": "Choose jurisdiction familiar to both parties or include neutral arbitration.",
    "confidentiality": "Clarify consequences and scope of confidential information."
}

# -----------------------------
# Function: Most Important Clauses (Get top 5 important clauses)
# -----------------------------
def get_top_clauses(clauses, embeddings, top_n=5):
    sim_matrix = cosine_similarity(embeddings)
    importance_scores = sim_matrix.sum(axis=1)
    top_indices = np.argsort(importance_scores)[-top_n:][::-1]
    return [(clauses[i], importance_scores[i]) for i in top_indices]

# -----------------------------
# Function: Clause Splitting
# -----------------------------
def split_into_clauses(text):
    # Merge numbered section headers with following text
    merged_text = re.sub(r'\n(\d+(\.\d+)?\s+[A-Z])', r' \1', text)
    raw_clauses = [c.strip() for c in merged_text.split('\n') if c.strip()]
    return raw_clauses

# -----------------------------
# Function: Read PDF text
# -----------------------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ----------------
# Main Processing
# ----------------
if uploaded_file is not None:
    # Read file depending on type
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        text = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.subheader("Document Preview:")
    st.write(text[:1000] + "..." if len(text) > 1000 else text)

    # Use improved clause splitting
    clauses = split_into_clauses(text)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode clauses
    clause_embeddings = model.encode(clauses)

    # Encode solution texts
    solution_texts = list(risk_solutions.values())
    solution_embeddings = model.encode(solution_texts)

    # -------------------------------------
    # Show Most Important Clauses (Summary)
    # -------------------------------------
    st.subheader("Most Important Clauses (Summary):")
    top_clauses = get_top_clauses(clauses, clause_embeddings, top_n=5)
    for clause, score in top_clauses:
        st.markdown(f"- {clause}")

    # ----------------------------------------------
    # Detect High-Risk Clauses and Show Suggestions
    # ----------------------------------------------
    st.subheader("Risk Analysis:")
    clf = IsolationForest(contamination=0.3, random_state=42)  # top 30% risky
    clf.fit(clause_embeddings)
    labels = clf.predict(clause_embeddings)  # -1 = risky, 1 = normal

    for clause, label, emb in zip(clauses, labels, clause_embeddings):
        if label == -1:  # HIGH RISK
            sims = cosine_similarity([emb], solution_embeddings)[0]
            best_idx = np.argmax(sims)
            suggestion = solution_texts[best_idx]
            st.markdown(
                f"<span style='color:red'>[HIGH RISK]</span> {clause} - Suggestion: {suggestion}",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<span style='color:green'>[LOW RISK]</span> {clause}", unsafe_allow_html=True)

# ---------------
# Notes / Footer
# ---------------
st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** The judgement-matching feature uses a small demo dataset of past judgements. "
    "For real legal research or predictions, you must integrate a comprehensive, licensed dataset of Indian judgments "
    "and consult a qualified lawyer. This app is a demo and not legal advice."
)
