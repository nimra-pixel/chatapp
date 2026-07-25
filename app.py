"""
ChatGPT-style app in a single Streamlit file.
- Streaming responses
- Multiple conversations saved to SQLite
- Works with any OpenAI-compatible API (Groq free by default)

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io (see README)
"""
import os
import json
import sqlite3
import uuid
from datetime import datetime

import httpx
import streamlit as st

import rag

# ---- Config ----------------------------------------------------------------
# On Streamlit Cloud these come from st.secrets; locally from env vars.
def cfg(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

API_KEY = cfg("LLM_API_KEY")
BASE_URL = cfg("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = cfg("LLM_MODEL", "llama-3.3-70b-versatile")
SYSTEM_PROMPT = cfg("SYSTEM_PROMPT", "You are a helpful, concise assistant.")
DB_PATH = cfg("DB_PATH", "chat.db")

st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")


# ---- Database --------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT, created_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT,
            content TEXT, created_at TEXT)""")
        rag.init_rag(conn)


def list_conversations():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title FROM conversations ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_history(cid):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at",
            (cid,)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(cid, role, content):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), cid, role, content, datetime.utcnow().isoformat()))


def create_conversation(first_message):
    cid = str(uuid.uuid4())
    title = (first_message[:38] + "…") if len(first_message) > 38 else first_message
    with get_conn() as conn:
        conn.execute("INSERT INTO conversations VALUES (?,?,?)",
                     (cid, title, datetime.utcnow().isoformat()))
    return cid


init_db()


# ---- Streaming call --------------------------------------------------------
def stream_completion(messages):
    """Yield text chunks from an OpenAI-compatible streaming endpoint."""
    payload = {"model": MODEL, "messages": messages, "stream": True}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.stream("POST", f"{BASE_URL}/chat/completions",
                      json=payload, headers=headers, timeout=None) as resp:
        if resp.status_code != 200:
            yield f"\n\n**Error {resp.status_code}:** {resp.read().decode()}"
            return
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta


# ---- State -----------------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("💬 Chat")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.conversation_id = None
        st.rerun()

    if not API_KEY:
        st.error("No API key set. Add LLM_API_KEY in secrets/env.")

    st.divider()

    # ---- Document upload (RAG) ----
    st.caption("📎 Chat with documents")
    uploads = st.file_uploader(
        "Upload files", type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploads:
        # Ensure a conversation exists to attach docs to
        cur_cid = st.session_state.conversation_id
        if cur_cid is None:
            cur_cid = create_conversation("Document chat")
            st.session_state.conversation_id = cur_cid
        with get_conn() as conn:
            existing = {s for s, _ in rag.list_documents(conn, cur_cid)}
            new_files = [f for f in uploads if f.name not in existing]
            if new_files:
                with st.spinner("Indexing documents…"):
                    for f in new_files:
                        n = rag.add_document(conn, cur_cid, f.name, f.getvalue())
                        st.success(f"{f.name}: {n} chunks indexed")
                st.rerun()

    # Show indexed docs for the active conversation
    if st.session_state.conversation_id:
        with get_conn() as conn:
            docs = rag.list_documents(conn, st.session_state.conversation_id)
        if docs:
            st.caption("Indexed:")
            for source, n in docs:
                st.caption(f"• {source} ({n})")

    st.divider()
    st.caption("Conversations")
    for c in list_conversations():
        active = c["id"] == st.session_state.conversation_id
        if st.button(("• " if active else "") + (c["title"] or "Untitled"),
                     key=c["id"], use_container_width=True):
            st.session_state.conversation_id = c["id"]
            st.rerun()
    st.divider()
    st.caption(f"Model: {MODEL}")


# ---- Main chat area --------------------------------------------------------
cid = st.session_state.conversation_id

# Render existing history
if cid:
    for m in get_history(cid):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
else:
    st.markdown("#### Start a conversation")
    st.caption("Type below. Your chats are saved in the sidebar.")

# Input
prompt = st.chat_input("Message…")
if prompt:
    if not API_KEY:
        st.stop()

    # Create conversation on first message
    if cid is None:
        cid = create_conversation(prompt)
        st.session_state.conversation_id = cid

    save_message(cid, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- RAG: retrieve relevant chunks if this conversation has documents ----
    system_content = SYSTEM_PROMPT
    with get_conn() as conn:
        hits = rag.retrieve(conn, cid, prompt, k=4)
    if hits:
        context = rag.build_context(hits)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nUse the following context from the user's documents to answer. "
            + "If the answer isn't in the context, say so and answer from general knowledge.\n\n"
            + "=== CONTEXT ===\n" + context + "\n=== END CONTEXT ==="
        )

    # Build full message list (model is stateless)
    messages = [{"role": "system", "content": system_content}] + get_history(cid)

    with st.chat_message("assistant"):
        full = st.write_stream(stream_completion(messages))
        if hits:
            with st.expander("📄 Sources used"):
                for score, source, content in hits:
                    st.caption(f"**{source}** (relevance {score:.2f})")
                    st.text(content[:300] + ("…" if len(content) > 300 else ""))

    save_message(cid, "assistant", full)
    st.rerun()
