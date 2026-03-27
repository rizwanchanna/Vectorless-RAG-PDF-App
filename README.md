# 📚 Vectorless RAG — Chat With Your PDFs

A Retrieval-Augmented Generation (RAG) app that lets you have a multi-turn conversation with your PDF documents — **without any vector database**. Instead of embeddings, it uses **TF-IDF + cosine similarity** for document retrieval, keeping the stack lightweight and dependency-free from heavy vector stores.

Built with [Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com), and powered by [OpenRouter](https://openrouter.ai) (Llama 3 8B by default).

---

## 🧠 How It Works

Traditional RAG pipelines rely on embedding models and vector databases (like FAISS, Chroma, Pinecone) to find relevant document chunks. This project replaces that entire layer with a classic NLP approach:

1. PDFs are loaded and split into chunks via `RecursiveCharacterTextSplitter`
2. Chunks are indexed using **TF-IDF** (`sklearn`)
3. At query time, **cosine similarity** ranks the most relevant chunks
4. Top chunks are passed as context to the LLM
5. A **history-aware retriever** rephrases follow-up questions into standalone queries
6. Conversation history is maintained per session

```
User Query
    │
    ▼
History-Aware Rephrasing (LLM)
    │
    ▼
TF-IDF Cosine Similarity Retrieval
    │
    ▼
Top-K Chunks → LLM (Llama 3 via OpenRouter)
    │
    ▼
Answer
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/rizwanchanna/Vectorless-RAG-PDF-App.git
cd Vectorless-RAG-PDF-App
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Also install the PDF loader dependency:
> ```bash
> pip install pypdf scikit-learn numpy
> ```

### 3. Get an OpenRouter API Key

Sign up at [https://openrouter.ai](https://openrouter.ai) and grab a free API key.

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Paste your **OpenRouter API Key** in the sidebar
3. Set a **Session ID** (used to isolate conversation history)
4. Upload one or more **PDF files**
5. Ask questions in the chat input — the app answers strictly from your documents

---

## 📁 Project Structure

```
Vectorless-RAG-PDF-App/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── temp.pdf            # Temporary file used during PDF processing
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| LLM Model | `meta-llama/llama-3-8b-instruct` | OpenRouter model used for answering |
| Chunk Size | `700` | Characters per document chunk |
| Chunk Overlap | `100` | Overlap between consecutive chunks |
| Top-K Retrieval | `2` | Number of chunks retrieved per query |
| Max Context Chars | `3000` | Character cap on context passed to LLM |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `langchain-classic` | RAG chain construction |
| `langchain-core` | Runnables, prompts, history |
| `langchain-community` | PDF loader, chat history |
| `langchain-openrouter` | OpenRouter LLM integration |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `numpy` | Array operations for similarity ranking |
| `pypdf` | PDF parsing |

---

## 💡 Why Vectorless?

| | Traditional RAG | Vectorless RAG |
|---|---|---|
| Retrieval | Embedding model + vector DB | TF-IDF + cosine similarity |
| Setup complexity | High | Low |
| Cost | Embedding API calls | Free (sklearn) |
| Accuracy | Higher (semantic) | Good for keyword-heavy queries |
| Best for | Large-scale production | Prototyping, lightweight apps |

---

## 🔒 Notes

- The app does **not** store your API key anywhere — it's only held in the current session
- Uploaded PDFs are temporarily written to `temp.pdf` and processed in-memory
- Each session ID maintains its own isolated conversation history
