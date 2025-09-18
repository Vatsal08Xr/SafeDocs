# ⚖️ Legal Document Analyzer

This project is a **Streamlit-based web app** that helps analyze legal documents.  
It can process `.txt` and `.pdf` files to detect risks, suggest improvements, and find related case law.

---

## ✨ Features
- 📑 **Clause Analysis** – Splits documents into clauses and highlights **high-risk** and **low-risk** parts.  
- 💡 **AI Suggestions** – Provides improvement tips for high-risk clauses.  
- 📌 **Important Clauses** – Extracts the most crucial parts of the document.    
- 📂 **PDF & TXT Support** – Works with both text files and scanned legal PDFs.  

---

### 🚀 How to Run Locally

### 1. Clone the repository
bash
git clone https://github.com/Vatsal08Xr/legal-doc-analyzer.git
cd legal-doc-analyzer


### 2. Create a venv
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the app

streamlit run legal.py
