# ChatGPT-style app — Streamlit version

One file, one deploy. Streaming replies, saved conversations, works with any
OpenAI-compatible API (Groq free by default).

Files:
- `app.py` — the whole app
- `requirements.txt` — dependencies

---

## Step 1 — Get a free API key (Groq recommended)
1. Sign up at https://console.groq.com
2. **API Keys → Create API Key** → copy it (starts with `gsk_`)
3. Free model already set as default: `llama-3.3-70b-versatile`

(Other options: Gemini — base URL `https://generativelanguage.googleapis.com/v1beta/openai`,
model `gemini-2.0-flash`. OpenRouter — base URL `https://openrouter.ai/api/v1`, any `:free` model.)

---

## Step 2 — Run locally
```bash
cd chatapp-streamlit
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export LLM_API_KEY="gsk_your_key_here"     # Windows PS: $env:LLM_API_KEY="gsk_..."
streamlit run app.py
```
Browser opens at http://localhost:8501. Start chatting.

To change the model or prompt without editing code:
```bash
export LLM_MODEL="llama-3.1-8b-instant"
export SYSTEM_PROMPT="You are an expert Urdu tutor."
```

---

## Step 3 — Deploy free on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo** (see commands below).
2. Go to https://share.streamlit.io → **Create app** → pick your repo.
   - **Main file path:** `app.py`
3. Before/after first deploy, open **⋮ → Settings → Secrets** and paste:
   ```toml
   LLM_API_KEY = "gsk_your_key_here"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   SYSTEM_PROMPT = "You are a helpful, concise assistant."
   ```
4. Save. The app redeploys and is live at `https://<your-app>.streamlit.app`.

Push to GitHub:
```bash
git init
git add .
git commit -m "Streamlit chat app"
git branch -M main
git remote add origin https://github.com/nimra-pixel/chatapp.git
git push -u origin main
```

> **Never commit your API key.** Keep it only in Streamlit **Secrets** (or local env vars).
> The app reads `st.secrets` first, then env vars.

> **Persistence note:** SQLite lives on the app's disk, which resets on redeploy/reboot.
> Fine for a demo/portfolio. For durable storage, swap to a hosted Postgres
> (e.g. Supabase free tier) later.

---

## Why Streamlit here
- One deploy instead of separate backend + frontend.
- No servers to manage; free hosting; instant public URL to share.
- You already work in Streamlit, so extending it (file upload, RAG, auth) is fast.

## Natural next steps
1. **RAG** — `st.file_uploader` → chunk → embed → retrieve → inject as context.
2. **Model picker** — a sidebar `st.selectbox` to switch models live.
3. **Postgres** — durable conversation storage.
4. **Auth** — `streamlit-authenticator` for per-user chats.
