"""
Lightweight RAG: extract text from uploads, chunk, embed, retrieve.
Embeddings use scikit-learn TF-IDF — no PyTorch, no model download, installs
fast on any Python version (including 3.14 on Streamlit Cloud).

The TF-IDF vectorizer is fit per conversation over that conversation's chunks
and cached in memory; vectors are recomputed from stored chunk text on load.
"""
import io
import re
import sqlite3
import uuid
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cache one fitted vectorizer + matrix per conversation, keyed by chunk count
# so it refits when new documents are added.
_cache = {}


# ---- Text extraction -------------------------------------------------------
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")


# ---- Chunking --------------------------------------------------------------
def chunk_text(text: str, size: int = 800, overlap: int = 150):
    """Split into ~size-char chunks, preferring paragraph boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= size:
                cur = p
            else:
                words, buf = p.split(), []
                for w in words:
                    buf.append(w)
                    if len(" ".join(buf)) >= size:
                        chunks.append(" ".join(buf))
                        buf = " ".join(buf)[-overlap:].split()
                cur = " ".join(buf)
    if cur:
        chunks.append(cur)
    return chunks


# ---- Storage ---------------------------------------------------------------
def init_rag(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            source TEXT,
            content TEXT,
            created_at TEXT
        )
    """)


def add_document(conn, conversation_id, filename, data):
    """Extract, chunk, and store. Returns number of chunks added."""
    text = extract_text(filename, data)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    now = datetime.utcnow().isoformat()
    rows = [
        (str(uuid.uuid4()), conversation_id, filename, chunk, now)
        for chunk in chunks
    ]
    conn.executemany(
        "INSERT INTO doc_chunks (id, conversation_id, source, content, created_at) VALUES (?,?,?,?,?)",
        rows)
    conn.commit()
    _cache.pop(conversation_id, None)  # invalidate so it refits with new chunks
    return len(chunks)


def list_documents(conn, conversation_id):
    rows = conn.execute(
        "SELECT source, COUNT(*) n FROM doc_chunks WHERE conversation_id=? GROUP BY source",
        (conversation_id,)).fetchall()
    return [(r[0], r[1]) for r in rows]


def _load_index(conn, conversation_id):
    """Fit (or reuse) a TF-IDF index over this conversation's chunks."""
    rows = conn.execute(
        "SELECT source, content FROM doc_chunks WHERE conversation_id=?",
        (conversation_id,)).fetchall()
    if not rows:
        return None
    sources = [r[0] for r in rows]
    contents = [r[1] for r in rows]

    cached = _cache.get(conversation_id)
    if cached and cached["n"] == len(contents):
        return cached

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(contents)
    index = {"n": len(contents), "vectorizer": vectorizer,
             "matrix": matrix, "sources": sources, "contents": contents}
    _cache[conversation_id] = index
    return index


def retrieve(conn, conversation_id, query, k=4):
    """Return the top-k most similar chunks to the query."""
    index = _load_index(conn, conversation_id)
    if index is None:
        return []
    q_vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(q_vec, index["matrix"])[0]
    order = np.argsort(sims)[::-1][:k]
    results = []
    for i in order:
        score = float(sims[i])
        # Keep any chunk with a nonzero match. If nothing matched at all
        # (query shares no vocabulary), fall through to returning the top
        # chunks anyway so the model still gets document context.
        results.append((score, index["sources"][i], index["contents"][i]))
    # Drop trailing all-zero results only if at least one real match exists.
    if any(r[0] > 0 for r in results):
        results = [r for r in results if r[0] > 0]
    return results


def build_context(chunks):
    if not chunks:
        return ""
    parts = []
    for i, (score, source, content) in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {source}]\n{content}")
    return "\n\n".join(parts)
