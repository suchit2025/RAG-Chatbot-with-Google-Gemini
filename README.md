# 🧠 RAG Chatbot with Google Gemini 

A Retrieval-Augmented Generation (RAG) chatbot powered by **Google Gemini 2.5 Flash**, designed to deliver intelligent, document-aware responses. Upload PDFs, embed them locally, and chat with context-aware answers — all through a clean Streamlit interface.

---

## 🚀 Features

- 🔗 **Gemini API Integration** — Uses Gemini 2.5 Flash for fast, accurate responses.
- 📄 **PDF Upload Support** — Drag-and-drop or browse files (up to 200MB).
- 🧩 **Configurable Chunking** — Tune chunk size and overlap for optimal embedding.
- 🧠 **Local Embeddings** — CPU-based embedding for lightweight testing.
- 💬 **Document-Aware Chat** — Ask questions directly based on uploaded content.
- 🛠️ **Streamlit Interface** — Simple, responsive UI for interaction and debugging.

---

## 📦 Example Use Case

Uploaded `samsung-mobile-all-model-list-492.pdf` and queried:
- “Tell me all mobile names” → Lists Galaxy models like J7, S6 Edge, Note 5, etc.
- “Tell me price one by one” → Indicates price data not available in the document.
- “Which models come under Galaxy Tab S3 9.7” → Returns model codes like SM-T813, SM-T818T, SM-T819.

---

## 🧰 Tech Stack

| Layer            | Tools & Libraries                     |
|------------------|----------------------------------------|
| LLM Backend      | Google Gemini 2.5 Flash (via API)      |
| Embedding Engine | SentenceTransformers (CPU)             |
| UI Framework     | Streamlit                              |
| File Handling    | PyMuPDF, LangChain Document Loaders    |
| Chunking & RAG   | LangChain Text Splitters + FAISS       |

---

## 🖼️ Architecture Overview

<img width="956" height="782" alt="image" src="https://github.com/user-attachments/assets/71813ae5-b29f-42c0-bc02-bfc27dbe5611" />

--

## 🧪 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/suchit2025/RAG-Chatbot-with-Google-Gemini.git
cd RAG-Chatbot-with-Google-Gemini
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
Create a `.env` file:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
RAG-Chatbot-with-Google-Gemini/
├── app.py
├── utils/
│   ├── embedder.py
│   ├── loader.py
│   └── chat_engine.py
├── .env
├── requirements.txt
└── README.md
```

---

## ✨ Credits

Built by [Suchit Gaikwad](https://github.com/suchit2025) — AI agent developer and automation engineer passionate about modular, recruiter-ready platforms.

---

## 📸 Screenshots 

<img width="1864" height="776" alt="image" src="https://github.com/user-attachments/assets/4d8d78ad-ecb6-4c8b-9587-dae68d8616e9" />
<img width="1866" height="719" alt="image" src="https://github.com/user-attachments/assets/07e07aa4-2a2a-4d02-a5f3-aecf5d55cab6" />


---


