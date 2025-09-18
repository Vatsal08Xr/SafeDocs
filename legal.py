# Step 0: Install dependencies if not already done
# pip install streamlit sentence-transformers scikit-learn numpy

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="Legal Document Analyzer with Suggestions", layout="wide")
st.title("📄 Legal Document Risk Analyzer with Suggestions & Summary")
st.write("Upload a legal document (.txt) to detect high-risk clauses, see suggestions, and summarize the most important clauses.")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("Upload a .txt legal document", type="txt")

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
# Function: Most Important Clauses
# -----------------------------
def get_top_clauses(clauses, embeddings, top_n=5):
    sim_matrix = cosine_similarity(embeddings)
    importance_scores = sim_matrix.sum(axis=1)
    top_indices = np.argsort(importance_scores)[-top_n:][::-1]
    return [(clauses[i], importance_scores[i]) for i in top_indices]

# -----------------------------
# Function: Clause Splitting (improved)
# -----------------------------
def split_into_clauses(text):
    """
    Improved clause splitter:
    - Keeps numbered titles like '2. SERVICES' with their following text
    - Splits cleanly on newlines, not just periods
    """
    # Merge numbered section headers with their following line
    merged_text = re.sub(r'\n(\d+(\.\d+)?\s+[A-Z])', r' \1', text)
    # Split on newlines
    raw_clauses = [c.strip() for c in merged_text.split('\n') if c.strip()]
    return raw_clauses

# -----------------------------
# Main Processing
# -----------------------------
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
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

    # -----------------------------
    # Show Most Important Clauses (Summary)
    # -----------------------------
    st.subheader("Most Important Clauses (Summary):")
    top_clauses = get_top_clauses(clauses, clause_embeddings, top_n=5)
    for clause, score in top_clauses:
        st.markdown(f"- {clause}")

    # -----------------------------
    # Detect High-Risk Clauses and Show Suggestions
    # -----------------------------
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
