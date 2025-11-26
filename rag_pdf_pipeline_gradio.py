
"""
Clean RAG PDF Chatbot - Stable GPT-3.5-turbo Version
---------------------------------------------------
- Uses LangChain ONLY for embeddings + Chroma vector DB
- Uses OpenAI directly for chat responses
- GPT-3.5-turbo ensures stable outputs
- All previous bugs removed
"""

import os
import hashlib
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

# OpenAI direct client (stable)
from openai import OpenAI
client = OpenAI()

# Gradio UI
import gradio as gr


# ---------------- CONFIG ----------------
PERSIST_DIR = "chroma_pdf_db"
DOCS_DIR = "uploaded_pdfs"

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"        # CLEAN, STABLE, SUPPORTED
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
TOP_K = 4
# ----------------------------------------

os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

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

    if empty > len(pages) // 2 and _HAS_PDFPLUMBER:
        fallback = extract_text_pdfplumber(path)
        if fallback:
            return fallback
    return pages

# Convert pages to LangChain docs
def pages_to_docs(filename: str, pages):
    docs = []
    for page_no, text in pages:
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": page_no}
            )
        )
    return docs


# Chunk PDF text
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# ---------- Save uploaded file ----------
def save_uploaded_file(uploaded, dest=DOCS_DIR):
    os.makedirs(dest, exist_ok=True)

    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return uploaded

    # Object with .name + .read()
    if hasattr(uploaded, "read"):
        data = uploaded.read()
        name = os.path.basename(uploaded.name)
        full = os.path.join(dest, name)
        with open(full, "wb") as f:
            f.write(data)
        return full

    raise ValueError("Unsupported upload type")


# ---------- Ingest PDF ----------
def ingest_pdf(path: str):
    filename = os.path.basename(path)
    pages = extract_pdf(path)

    docs = pages_to_docs(filename, pages)
    if not docs:
        return f"No text in {filename}"

    chunks = chunk_documents(docs)

    embeddings_local = OpenAIEmbeddings(model=EMBED_MODEL)

    # Build Chroma DB
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_local,
            persist_directory=PERSIST_DIR
        )
    except:
        vect = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_local
        )
        vect.add_documents(chunks)

    return f"Ingested {len(chunks)} chunks from {filename}"


# ---------- Load vector DB at startup ----------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
try:
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
except:
    vectordb = None

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K}) if vectordb else None


# ---------- LLM Call (GPT-3.5-turbo) ----------
def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0,
        )
        msg = response.choices[0].message
        if hasattr(msg, "content"):
            return msg.content.strip()
        # fallback (older/newer structures)
        try:
            return msg["content"].strip()
        except Exception:
            return str(msg).strip()
    except Exception as e:
        return f"LLM Error: {e}"

# ---------- RAG Answer ----------
def answer_question(query: str, history):

    # 1) Retrieve (safe for all LangChain versions)
    if retriever:
        # Try normal method
        try:
            docs = retriever.get_relevant_documents(query)
        except:
            docs = []

        # Try fallback method (LangChain newer versions)
        if not docs:
            try:
                docs = retriever._get_relevant_documents(query, run_manager=None)
            except:
                docs = []
    else:
        docs = []

    # 2) Build context
    context = "\n\n".join(d.page_content for d in docs) if docs else ""

    if context.strip():
        prompt = f"""
Use the following context to answer the question.

If the answer is NOT found in the context,
answer using your own general knowledge.

Context:
{context}

Question: {query}
"""
    else:
        # No context → allow normal ChatGPT-style answer
        prompt = f"""
Answer the following question using your own knowledge:

{query}
"""




    # 4) LLM answer
    answer = call_llm(prompt)

    # 5) Sources
    src_lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        snippet = d.page_content[:150].replace("\n", " ")
        src_lines.append(f"{i}. {meta.get('source')} (page {meta.get('page')}) — {snippet}...")

    return answer, "\n".join(src_lines)

    
    



# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## 📘 Clean RAG PDF Chatbot (GPT-3.5-Turbo)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            qbox = gr.Textbox(label="Ask a question")
            ask_btn = gr.Button("Ask")
            clear_btn = gr.Button("Clear")
            sources_out = gr.Textbox(label="Sources", lines=10)

        with gr.Column(scale=1):
            upload = gr.File(label="Upload PDFs", file_count="multiple")
            ingest_btn = gr.Button("Ingest PDFs")
            status = gr.Textbox(label="Status")

    chat_state = gr.State([])

    def ask_fn(q, history):
        if not q.strip():
            return [], history, "Enter a question"

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

        msgs = []
        for f in files:
            saved = save_uploaded_file(f)
            msg = ingest_pdf(saved)
            msgs.append(msg)

        # Reload DB
        global vectordb, retriever
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        return "\n".join(msgs)

    ask_btn.click(ask_fn, inputs=[qbox, chat_state], outputs=[chatbot, chat_state, sources_out])
    clear_btn.click(lambda: ([], [], ""), outputs=[chatbot, chat_state, sources_out])
    ingest_btn.click(ingest_fn, inputs=[upload], outputs=[status])

demo.launch()
