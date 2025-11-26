"""
Final Clean RAG PDF Chatbot (GPT-3.5-turbo) with:
 - improved chunking (smaller chunks)
 - increased TOP_K retrieval
 - auto-ingest on upload
 - friendly small-talk bypass
 - SAFE: uses a fresh persist dir per ingest (no risky deletes)
"""

import os
import time
import hashlib
import re
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Load .env for OpenAI key
load_dotenv()

# PDF extraction
from pypdf import PdfReader
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except:
    _HAS_PDFPLUMBER = False

# LangChain components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# OpenAI direct client
from openai import OpenAI
client = OpenAI()

# Gradio UI
import gradio as gr

# ---------------- CONFIG ----------------
PERSIST_DIR = "chroma_pdf_db"   # parent prefix for persist dirs; actual dirs will be PERSIST_DIR_<ts>
DOCS_DIR = "uploaded_pdfs"

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
TOP_K = 6
MAX_TOKENS = 800
# ----------------------------------------

os.makedirs(DOCS_DIR, exist_ok=True)

# Current active persist dir (will be set after first ingest)
CURRENT_PERSIST_DIR = None

# ---------- Helpers ----------
def make_persist_dir():
    """Create and return a unique persist dir for each ingest."""
    ts = int(time.time() * 1000)
    new_dir = f"{PERSIST_DIR}_{ts}"
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    return new_dir

# ---------- PDF extraction ----------
def extract_text_pypdf(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except:
            text = ""
        pages.append((i + 1, text))
    return pages

def extract_text_pdfplumber(path: str):
    if not _HAS_PDFPLUMBER:
        return []
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            try:
                text = p.extract_text() or ""
            except:
                text = ""
            pages.append((i + 1, text))
    return pages

def extract_pdf(path: str):
    pages = extract_text_pypdf(path)
    empty = sum(1 for (_, t) in pages if not t.strip())
    if empty > len(pages)//2 and _HAS_PDFPLUMBER:
        fb = extract_text_pdfplumber(path)
        if fb:
            return fb
    return pages

# Convert pages to LangChain docs
def pages_to_docs(filename: str, pages: List[Tuple[int, str]]):
    docs = []
    for page_no, text in pages:
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={"source": filename, "page": page_no}))
    return docs

# Chunk PDF text
def chunk_documents(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

# ---------- Save uploaded file ----------
def save_uploaded_file(uploaded, dest=DOCS_DIR):
    os.makedirs(dest, exist_ok=True)

    # if gradio returns a path string
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return uploaded

    # file-like object with .name and .read
    if hasattr(uploaded, "read"):
        data = uploaded.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        name = os.path.basename(getattr(uploaded, "name", "uploaded.pdf"))
        path = os.path.join(dest, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    raise ValueError("Unsupported upload type")

# ---------- Ingest (SAFE: new persist dir per ingest) ----------
def ingest_pdf(path: str):
    filename = os.path.basename(path)

    pages = extract_pdf(path)
    docs = pages_to_docs(filename, pages)
    if not docs:
        return f"No text found in {filename}"

    chunks = chunk_documents(docs)
    embeddings_local = OpenAIEmbeddings(model=EMBED_MODEL)

    # create a new persist directory for this ingest
    new_persist = make_persist_dir()

    # Build Chroma DB in the new directory
    try:
        Chroma.from_documents(documents=chunks, embedding=embeddings_local, persist_directory=new_persist)
    except Exception:
        # fallback
        vect = Chroma(persist_directory=new_persist, embedding_function=embeddings_local)
        vect.add_documents(chunks)

    # Reinitialize global vectordb/retriever to point to this new directory
    global embeddings, vectordb, retriever, CURRENT_PERSIST_DIR
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    CURRENT_PERSIST_DIR = new_persist
    vectordb = Chroma(persist_directory=CURRENT_PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    return f"Ingested {len(chunks)} chunks from {filename} into {CURRENT_PERSIST_DIR}"

# ---------- Load DB at startup (no initial DB) ----------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectordb = None
retriever = None
# if you want to load an existing persist dir at startup, set CURRENT_PERSIST_DIR and init below.

# ---------- SAFE RETRIEVAL ----------
def safe_retrieve(query: str, k=TOP_K):
    global retriever, vectordb
    if retriever:
        try:
            # prefer public API if available
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query)
                if docs:
                    return docs
        except Exception:
            pass
        # try private api
        try:
            if hasattr(retriever, "_get_relevant_documents"):
                try:
                    return retriever._get_relevant_documents(query, run_manager=None)
                except TypeError:
                    return retriever._get_relevant_documents(query)
        except Exception:
            pass

    if vectordb:
        try:
            if hasattr(vectordb, "similarity_search"):
                return vectordb.similarity_search(query, k=k)
            if hasattr(vectordb, "search"):
                return vectordb.search(query, k=k)
        except Exception:
            pass

    return []

# ---------- LLM ----------
def call_llm(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        msg = res.choices[0].message
        if hasattr(msg, "content"):
            return msg.content.strip()
        try:
            return msg["content"].strip()
        except Exception:
            return str(msg).strip()
    except Exception as e:
        return f"LLM Error: {e}"

# ---------- ANSWER FUNCTION ----------
def answer_question(query: str, history):
    # Small-talk bypass
    s = (query or "").lower().strip()
    if s in ["hi", "hello", "hey", "whats up", "what's up", "how are you", "how r u"]:
        return call_llm(f"Reply casually: {query}"), "General knowledge"

    # Retrieve relevant chunks
    docs = safe_retrieve(query, k=TOP_K)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    if not context.strip():
        # Normal chatbot mode
        return call_llm(query), "No PDF context used"

    # PDF-based answer
    prompt = f"""
Use the following context to answer the question.
If context does not contain the answer, answer normally.

Context:
{context}

Question: {query}
"""
    return call_llm(prompt), "PDF context used"

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## 📘 Final RAG Chatbot (GPT-3.5) — Answers Only From Latest PDF (safe)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420)
            qbox = gr.Textbox(label="Ask")
            ask_btn = gr.Button("Ask")
            clear_btn = gr.Button("Clear Chat")
            sources_out = gr.Textbox(label="Sources", lines=6)

        with gr.Column(scale=1):
            upload = gr.File(label="Upload PDF", file_count="multiple")
            ingest_btn = gr.Button("Ingest PDF")
            status = gr.Textbox(label="Status")

    chat_state = gr.State([])

    def ask_fn(q, history):
        ans, src = answer_question(q, history)
        history.append((q, ans))
        msgs = []
        for u, a in history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        return msgs, history, src

    def ingest_fn(files):
        if not files:
            return "No files"

        file_list = files if isinstance(files, list) else [files]
        msgs = []

        for f in file_list:
            saved = save_uploaded_file(f)
            msgs.append(ingest_pdf(saved))

        return "\n".join(msgs)

    # Auto-ingest on upload
    upload.change(ingest_fn, inputs=[upload], outputs=[status])

    ask_btn.click(ask_fn, inputs=[qbox, chat_state], outputs=[chatbot, chat_state, sources_out])
    clear_btn.click(lambda: ([], [], ""), outputs=[chatbot, chat_state, sources_out])
    ingest_btn.click(ingest_fn, inputs=[upload], outputs=[status])

if __name__ == "__main__":
    demo.launch()
