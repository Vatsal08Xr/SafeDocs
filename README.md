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

## 🚀 Installation

1. Clone the repository

```bash
git clone https://github.com/Vatsal08Xr/SaiVatsal_11.git
cd SaiVatsal_11
```

2. Create a virtual environment

   On **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   On **Mac/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run legal.py
```

5. Done!
