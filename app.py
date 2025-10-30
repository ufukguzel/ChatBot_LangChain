import os
import sys
import io
import uuid
import argparse
from typing import Dict, List, Optional, Tuple

"""
LangChain Chatbot — CLI + Web UI (FastAPI) with Streaming & RAG (+ PDF + Clickable Sources)
-------------------------------------------------------------------------------------------
Features:
  1) Dependency-free CLI chatbot (mock-capable)
  2) FastAPI Web UI (no Streamlit) — improved UI, theme toggle, copy/regen, autosize
  3) **Streaming responses** (chunked HTTP)
  4) **File upload & RAG** (TXT/MD/**PDF**); cited sources with **click-to-open modal**

The app degrades gracefully:
- If LangChain/OpenAI packages or API key are missing, it falls back to a deterministic mock.
- If FAISS/Embeddings/Text splitters are missing, it uses a simple keyword retriever.
- If PDF lib missing, falls back to naive bytes decode.

Usage
-----
CLI (no extra deps):
  $ python app.py                   # interactive CLI
  $ python app.py --mock            # force mock mode
  $ python app.py --run-tests       # run unit tests

Web UI (requires extra deps):
  $ pip install fastapi uvicorn
  $ python app.py --api --host 0.0.0.0 --port 8000
  # open http://localhost:8000

Optional (for better RAG):
  $ pip install langchain-text-splitters faiss-cpu langchain-openai langchain-core
  $ pip install python-dotenv pypdf

Env
----
OPENAI_API_KEY is read from .env (if python-dotenv is available) or from the environment.
Set FORCE_MOCK=1 to force mock mode.
"""

# --- Optional dotenv ---
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv() -> None:  # fallback no-op
        return None

load_dotenv()

# --- Try LangChain imports, fall back gracefully ---
_LC_AVAILABLE = True
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
    from langchain_core.runnables.history import RunnableWithMessageHistory  # type: ignore
    from langchain_core.chat_history import InMemoryChatMessageHistory  # type: ignore
except Exception:
    _LC_AVAILABLE = False

# --- Try FastAPI (optional) ---
_API_AVAILABLE = True
try:
    from fastapi import FastAPI, Request, UploadFile, File, Query
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except Exception:
    _API_AVAILABLE = False

# --- Optional RAG helpers ---
_SPLITTER_AVAILABLE = True
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    _SPLITTER_AVAILABLE = False

_FAISS_AVAILABLE = True
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    _FAISS_AVAILABLE = False

# --- Optional PDF ---
_PDF_AVAILABLE = True
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    _PDF_AVAILABLE = False

# ------------------------------
# Minimal mock & utilities
# ------------------------------
class SimpleHistory:
    """Framework-agnostic in-memory chat history."""
    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_ai(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def as_text(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)


class MockLLM:
    """Deterministic mock model that echoes with lightweight transformation.
    Keeps tests offline and predictable.
    """
    def __init__(self, style: str = "succinct") -> None:
        self.style = style

    def stream(self, text: str):
        tokens = text.split()
        for t in tokens:
            yield t + " "

    def invoke(self, prompt_text: str) -> str:
        last_line = prompt_text.strip().splitlines()[-1] if prompt_text.strip() else ""
        user_part = last_line.replace("user:", "").strip()
        return f"MOCK[{self.style}]: {user_part}"


# ------------------------------
# Simple RAG store (with IDs + FAISS metadata)
# ------------------------------
class InMemoryDoc:
    def __init__(self, doc_id: str, text: str, source: str):
        self.id = doc_id
        self.text = text
        self.source = source

class SimpleKeywordRetriever:
    """Very small fallback retriever using keyword overlap scoring."""
    def __init__(self, docs: List[InMemoryDoc]):
        self.docs = docs

    @staticmethod
    def _score(q: str, text: str) -> int:
        qs = {w.lower() for w in q.split() if len(w) > 2}
        ts = {w.lower() for w in text.split() if len(w) > 2}
        return len(qs & ts)

    def get_relevant(self, query: str, k: int = 3) -> List[InMemoryDoc]:
        ranked = sorted(self.docs, key=lambda d: self._score(query, d.text), reverse=True)
        return ranked[:k]


# Global RAG state
_DOCS: List[InMemoryDoc] = []
_DOC_INDEX: Dict[str, InMemoryDoc] = {}
_FAISS_INDEX = None  # type: ignore
_EMB = None  # type: ignore


def _split_text(text: str) -> List[str]:
    if _SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        return [c.page_content for c in splitter.create_documents([text])]
    # naive fallback
    return [text[i:i+700] for i in range(0, len(text), 700)] or [text]


def _ensure_faiss(api_key: Optional[str]) -> bool:
    global _EMB
    if not _FAISS_AVAILABLE or not api_key:
        return False
    if _EMB is None:
        _EMB = OpenAIEmbeddings(api_key=api_key)
    return True


def _read_pdf_bytes(data: bytes) -> str:
    if not _PDF_AVAILABLE:
        return data.decode(errors="ignore")
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception:
        return data.decode(errors="ignore")


def add_documents(name: str, raw: bytes, api_key: Optional[str]) -> int:
    """Add uploaded content to the store; returns number of chunks added."""
    lower = (name or "").lower()
    if lower.endswith(".pdf"):
        content = _read_pdf_bytes(raw)
    else:
        try:
            content = raw.decode(errors="ignore")
        except Exception:
            content = ""

    chunks = _split_text(content)

    metadatas = []
    texts = []
    for ch in chunks:
        doc_id = str(uuid.uuid4())
        doc = InMemoryDoc(doc_id, ch, source=name)
        _DOCS.append(doc)
        _DOC_INDEX[doc_id] = doc
        texts.append(ch)
        metadatas.append({"source": name, "id": doc_id})

    if _ensure_faiss(api_key) and texts:
        global _FAISS_INDEX
        if _FAISS_INDEX is None:
            _FAISS_INDEX = FAISS.from_texts(texts, _EMB, metadatas=metadatas)
        else:
            _FAISS_INDEX.add_texts(texts, metadatas=metadatas)
    return len(chunks)


def retrieve(query: str, api_key: Optional[str], k: int = 4) -> List[Tuple[str, str, str]]:
    """Return list of (id, chunk_text, source)."""
    if _FAISS_INDEX is not None and _EMB is not None and api_key:
        results = _FAISS_INDEX.similarity_search(query, k=k)
        out = []
        for r in results:
            rid = str(r.metadata.get("id") or "")
            src = r.metadata.get("source", "unknown")
            txt = r.page_content
            out.append((rid, txt, src))
        return out
    # fallback keyword retriever
    retr = SimpleKeywordRetriever(_DOCS)
    docs = retr.get_relevant(query, k=k)
    return [(d.id, d.text, d.source) for d in docs]


# ------------------------------
# Chat Engine
# ------------------------------
class ChatEngine:
    """Unified interface over LangChain (if available) or a mock pipeline."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str],
        session_id: Optional[str] = None,
        force_mock: bool = False,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._use_langchain = _LC_AVAILABLE and (not force_mock) and bool(api_key)
        self.history_map: Dict[str, object] = {}
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

        if self._use_langchain:
            llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
            system_prompt = (
                "Sen deneyimli bir yardımcı asistansın. Yanıtlarında kısa, net ve doğru ol. "
                "Kullanıcı Türkçe konuşuyorsa Türkçe yanıt ver. Gerekirse adım adım açıkla."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("system", "Bağlam (varsa):\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            base_chain = prompt | llm | StrOutputParser()

            def _get_session_history(sid: str):
                if sid not in self.history_map:
                    self.history_map[sid] = InMemoryChatMessageHistory()
                return self.history_map[sid]

            self._chain = RunnableWithMessageHistory(
                base_chain,
                _get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
        else:
            self._mock = MockLLM()
            self._histories: Dict[str, SimpleHistory] = {}

    def _get_simple_history(self, sid: str) -> SimpleHistory:
        if sid not in self._histories:
            self._histories[sid] = SimpleHistory()
        return self._histories[sid]

    def reset(self) -> None:
        if hasattr(self, "_histories"):
            self._histories.pop(self.session_id, None)
        if hasattr(self, "history_map"):
            self.history_map.pop(self.session_id, None)

    def send(self, user_input: str, context: str = "") -> str:
        if self._use_langchain:
            result = self._chain.invoke(
                {"input": user_input, "context": context},
                config={"configurable": {"session_id": self.session_id}},
            )
            return str(result)
        else:
            h = self._get_simple_history(self.session_id)
            h.add_user(user_input)
            prompt_text = h.as_text() + ("\nCTX: " + context if context else "") + "\nuser: " + user_input
            reply = self._mock.invoke(prompt_text)
            h.add_ai(reply)
            return reply

    def stream(self, user_input: str, context: str = ""):
        if self._use_langchain:
            for chunk in self._chain.stream(
                {"input": user_input, "context": context},
                config={"configurable": {"session_id": self.session_id}},
            ):
                yield str(chunk)
        else:
            prefix = "MOCK[succinct]: "
            for ch in self._mock.stream(prefix + user_input):
                yield ch


# Global engine registry for Web UI sessions
_ENGINES: Dict[str, ChatEngine] = {}

def _get_engine_for_session(session_id: str, *, force_mock: bool, model: str, temperature: float) -> ChatEngine:
    api_key = os.getenv("OPENAI_API_KEY")
    if session_id not in _ENGINES:
        _ENGINES[session_id] = ChatEngine(
            model_name=model,
            temperature=temperature,
            api_key=api_key,
            session_id=session_id,
            force_mock=force_mock or (api_key is None),
        )
    return _ENGINES[session_id]


# ------------------------------
# CLI
# ------------------------------

def run_cli(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    engine = ChatEngine(
        model_name=args.model,
        temperature=args.temperature,
        api_key=api_key,
        force_mock=args.mock or (api_key is None),
    )

    print("🤖 LangChain Chatbot (CLI) — type /exit to quit\n")
    if not _LC_AVAILABLE:
        print("[i] LangChain paketleri bulunamadı — mock modda çalışıyor.")
    elif not api_key and not args.mock:
        print("[i] OPENAI_API_KEY yok — mock modda çalışıyor.")
    elif args.mock:
        print("[i] Zorunlu mock modu etkin.")

    while True:
        try:
            user = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor…")
            break
        if user in {"/exit", ":q", "quit"}:
            print("Görüşürüz! 👋")
            break
        if not user:
            continue
        reply = engine.send(user)
        print(f"🤖 Bot: {reply}\n")


# ------------------------------
# Web API (FastAPI) — improved UI + streaming + upload + clickable sources
# ------------------------------

def run_api(args: argparse.Namespace) -> None:
    if not _API_AVAILABLE:
        print("[!] FastAPI/Uvicorn bulunamadı. `pip install fastapi uvicorn` ile kurun.")
        sys.exit(1)

    app = FastAPI(title="LangChain Chatbot API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    INDEX_HTML = """
    <!doctype html>
    <html lang=\"tr\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>LangChain Chatbot</title>
      <style>
        :root { --bg:#0b1020; --panel:#0f1834; --muted:#9aa6d1; --accent:#5b8cff; --accent2:#00d1b2; --danger:#ff5b6e; --text:#eef1f6; }
        :root.light { --bg:#f6f7fb; --panel:#ffffff; --muted:#6b7280; --accent:#3b82f6; --accent2:#10b981; --danger:#ef4444; --text:#111827; }
        * { box-sizing: border-box; }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:0; background:var(--bg); color:var(--text); transition: background .2s ease, color .2s ease; }
        .wrap { max-width: 1080px; margin: 24px auto; padding: 16px; }

        .toolbar { display:flex; gap:8px; align-items:center; flex-wrap: wrap; padding: 12px; background: var(--panel); border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.15); position: sticky; top: 12px; z-index: 5; }
        .brand { font-weight: 800; letter-spacing: .2px; }
        .pill { border:1px solid #2a355a33; border-radius: 12px; padding:8px 10px; background:transparent; color:inherit; }
        select.pill, input.pill { height: 38px; }
        .btn { cursor:pointer; border:none; border-radius: 12px; padding:10px 12px; background: var(--accent); color:white; }
        .btn.secondary { background: #26407a; }
        .btn.ghost { background: transparent; border:1px solid #2a355a33; color:inherit; }
        .btn.danger { background: var(--danger); }
        .icon { width: 18px; height: 18px; display:inline-block; vertical-align:middle; }
        .grow { flex:1; }
        .right { margin-left:auto; display:flex; gap:8px; align-items:center; }
        .switch { display:flex; align-items:center; gap:8px; }

        .card { background: var(--panel); border-radius: 16px; padding: 0; box-shadow: 0 10px 30px rgba(0,0,0,.15); margin-top: 16px; min-height: 70vh; display:flex; flex-direction: column; }
        .chat { padding: 16px; display:flex; flex-direction:column; gap:10px; overflow:auto; height: 60vh; scroll-behavior: smooth; }
        .msg { padding:12px 14px; border-radius:16px; width: fit-content; max-width: 80%; line-height:1.55; position: relative; }
        .msg pre { margin: 8px 0; padding: 10px; border-radius: 12px; background: #00000022; overflow:auto; }
        .u { margin-left:auto; background:#26407a; color:#eef1f6; }
        .a { background:#1b2548; color:#eef1f6; }
        :root.light .u { background:#dbeafe; color:#111827; }
        :root.light .a { background:#f3f4f6; color:#111827; }
        .time { position:absolute; bottom:-18px; right:8px; font-size:11px; color: var(--muted); }
        .row { display:flex; gap:8px; padding: 12px; border-top:1px solid #1e2a5233; }
        textarea { flex:1; resize: none; min-height: 46px; max-height: 200px; border-radius:12px; border:1px solid #2a355a33; background:transparent; color:inherit; padding:10px 12px; }
        textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(91,140,255,.25); }

        .muted { color: var(--muted); font-size:12px; }
        .status { margin-left: 8px; }
        .spinner { width:14px; height:14px; border-radius:50%; border:2px solid #2a355a66; border-top-color: var(--accent2); display:inline-block; animation: spin 1s linear infinite; vertical-align:middle; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .toast { position: fixed; bottom: 16px; right: 16px; background:var(--panel); border:1px solid #2a355a33; padding:10px 12px; border-radius:12px; display:none; }
        .hidden { display:none; }
        .fab { position: fixed; right: 18px; bottom: 80px; border-radius: 999px; padding: 10px 12px; box-shadow: 0 8px 20px rgba(0,0,0,.2); }
        .bubble-tools { position:absolute; top:6px; right:6px; display:flex; gap:6px; opacity:0; transition:opacity .15s ease; }
        .msg:hover .bubble-tools { opacity: .9; }
        .tool { border:none; padding:6px 8px; font-size:11px; border-radius:8px; background:#00000033; color:#fff; cursor:pointer; }
        :root.light .tool { background:#00000010; color:#111; }
        .sources { padding: 10px 12px; border-top:1px dashed #2a355a55; }
        .sources a { color: var(--accent2); text-decoration: none; cursor: pointer; }

        /* Modal */
        .modal { position: fixed; inset: 0; display:none; align-items:center; justify-content:center; background: rgba(0,0,0,.4); }
        .modal .box { width: min(90vw, 820px); max-height: 80vh; overflow:auto; background: var(--panel); border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.4); }
        .modal .head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
      </style>
    </head>
    <body>
      <div class=\"wrap\">
        <div class=\"toolbar\">
          <div class=\"brand\">🤖 LangChain Chatbot</div>
          <span class=\"muted status\" id=\"status\"></span>
          <div class=\"right\">
            <div class=\"switch\">
              <label class=\"muted\">Tema</label>
              <button class=\"btn ghost\" id=\"theme\" title=\"Açık/Koyu\">🌗</button>
            </div>
            <select class=\"pill\" id=\"model\">
              <option value=\"gpt-4o-mini\" selected>gpt-4o-mini</option>
              <option value=\"gpt-4o\">gpt-4o</option>
            </select>
            <input class=\"pill\" id=\"temperature\" type=\"range\" min=\"0\" max=\"1.5\" step=\"0.1\" value=\"0.4\" style=\"width:160px\" title=\"Temperature\" />
            <span class=\"muted\" id=\"tval\">0.4</span>
            <label class=\"pill\" style=\"display:flex;align-items:center;gap:6px;\"><input type=\"checkbox\" id=\"use_rag\" /> Kaynakları kullan</label>
            <label class=\"pill\" style=\"display:flex;align-items:center;gap:6px;\"><input type=\"checkbox\" id=\"use_stream\" checked/> Akışlı yanıt</label>
            <button class=\"btn ghost\" id=\"clear\">Sohbeti Temizle</button>
            <button class=\"btn secondary\" id=\"export\">Dışa Aktar</button>
          </div>
        </div>

        <div class=\"card\">
          <div class=\"chat\" id=\"chat\"></div>
          <div class=\"row\" style=\"border-bottom:1px solid #1e2a5233\">
            <form id=\"uploader\" enctype=\"multipart/form-data\" style=\"display:flex;gap:8px;align-items:center;\">
              <input type=\"file\" name=\"file\" id=\"file\" class=\"pill\" />
              <button type=\"submit\" class=\"btn ghost\">Yükle</button>
              <span class=\"muted\" id=\"docinfo\"></span>
            </form>
          </div>
          <div class=\"row\">
            <textarea id=\"prompt\" placeholder=\"Bir şey yaz...\" rows=\"1\"></textarea>
            <button class=\"btn\" id=\"send\">Gönder</button>
          </div>
          <div class=\"sources hidden\" id=\"sources\"></div>
        </div>
        <p class=\"muted\">Not: API anahtarı yoksa otomatik mock moda geçilir. Ctrl/Cmd+Enter: Gönder</p>
      </div>

      <button class=\"btn fab\" id=\"scrolldown\">▼</button>
      <div class=\"toast\" id=\"toast\"></div>

      <div class=\"modal\" id=\"modal\"> 
        <div class=\"box\">
          <div class=\"head\">
            <div id=\"msource\" class=\"muted\"></div>
            <div>
              <button class=\"btn ghost\" id=\"copySrc\">Kopyala</button>
              <button class=\"btn danger\" id=\"close\">Kapat</button>
            </div>
          </div>
          <pre class=\"mono\"><code id=\"mtext\"></code></pre>
        </div>
      </div>

      <script>
        const chat = document.getElementById('chat');
        const promptEl = document.getElementById('prompt');
        const modelEl = document.getElementById('model');
        const tempEl = document.getElementById('temperature');
        const tval = document.getElementById('tval');
        const statusEl = document.getElementById('status');
        const sendBtn = document.getElementById('send');
        const clearBtn = document.getElementById('clear');
        const exportBtn = document.getElementById('export');
        const themeBtn = document.getElementById('theme');
        const scrollBtn = document.getElementById('scrolldown');
        const toast = document.getElementById('toast');
        const uploader = document.getElementById('uploader');
        const docinfo = document.getElementById('docinfo');
        const useRag = document.getElementById('use_rag');
        const useStream = document.getElementById('use_stream');
        const sourcesBox = document.getElementById('sources');
        const modal = document.getElementById('modal');
        const mtext = document.getElementById('mtext');
        const msource = document.getElementById('msource');
        const closeBtn = document.getElementById('close');
        const copySrc = document.getElementById('copySrc');

        function save(k,v){ try { localStorage.setItem(k, v); } catch {} }
        function load(k, d=null){ try { return localStorage.getItem(k) ?? d; } catch { return d; } }

        function applyTheme(name){
          if(name === 'light'){ document.documentElement.classList.add('light'); }
          else { document.documentElement.classList.remove('light'); }
          save('theme', name);
        }
        applyTheme(load('theme', 'dark'));
        themeBtn.addEventListener('click', ()=>{
          const now = document.documentElement.classList.contains('light') ? 'dark' : 'light';
          applyTheme(now);
        });

        const savedModel = load('model'); if(savedModel) modelEl.value = savedModel;
        modelEl.addEventListener('change', ()=> save('model', modelEl.value));
        const savedTemp = load('temp'); if(savedTemp){ tempEl.value = savedTemp; }
        tval.textContent = tempEl.value;
        tempEl.addEventListener('input', ()=> { tval.textContent = tempEl.value; save('temp', tempEl.value); });
        useRag.checked = load('use_rag', 'false') === 'true';
        useStream.checked = load('use_stream', 'true') === 'true';
        useRag.addEventListener('change', ()=> save('use_rag', useRag.checked));
        useStream.addEventListener('change', ()=> save('use_stream', useStream.checked));

        function showToast(text){ toast.textContent = text; toast.style.display = 'block'; setTimeout(()=> toast.style.display = 'none', 1800); }
        function ts(){ const d = new Date(); return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }

        function md(text){
          let safe = (text||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
          const parts = safe.split('```');
          let html = '';
          for(let i=0;i<parts.length;i++){
            if(i%2===1){ html += '<pre><code>'+parts[i]+'</code></pre>'; }
            else { let p = parts[i]
              .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
              .replace(/\*([^*]+)\*/g,'<em>$1</em>')
              .replace(/`([^`]+)`/g,'<code>$1</code>')
              .replace(/\n/g,'<br>'); html += p; }
          }
          return html;
        }

        function bubbleTools(){
          const c = document.createElement('div'); c.className = 'bubble-tools';
          const copy = document.createElement('button'); copy.className = 'tool'; copy.textContent = 'Kopyala';
          copy.addEventListener('click', (e)=>{ e.stopPropagation(); const text = c.parentElement.querySelector('.content').innerText; navigator.clipboard?.writeText(text); showToast('Kopyalandı'); });
          const regen = document.createElement('button'); regen.className = 'tool'; regen.textContent = 'Yenile';
          regen.addEventListener('click', (e)=>{ e.stopPropagation(); if(lastUser) send(lastUser, true); });
          c.appendChild(copy); c.appendChild(regen); return c;
        }

        function append(role, text){
          const wrap = document.createElement('div'); wrap.className = 'msg ' + (role === 'user' ? 'u' : 'a');
          const content = document.createElement('div'); content.className = 'content'; content.innerHTML = md(text);
          const time = document.createElement('div'); time.className = 'time'; time.textContent = ts();
          wrap.appendChild(content); wrap.appendChild(time); if(role === 'assistant') wrap.appendChild(bubbleTools());
          chat.appendChild(wrap); chat.scrollTop = chat.scrollHeight; return wrap;
        }

        function setLoading(v){ if(v){ statusEl.innerHTML = '<span class=\"spinner\"></span> yazıyor…'; sendBtn.disabled = true; promptEl.disabled = true; } else { statusEl.textContent = ''; sendBtn.disabled = false; promptEl.disabled = false; promptEl.focus(); } }

        async function ensureSession(){ try { await fetch('/session', { method:'POST' }); } catch {} }
        async function resetSession(){ try { await fetch('/reset', { method:'POST' }); } catch {} chat.innerHTML = ''; showToast('Sohbet temizlendi'); sourcesBox.classList.add('hidden'); sourcesBox.innerHTML=''; }

        function autosize(){ promptEl.style.height = 'auto'; promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + 'px'; }
        promptEl.addEventListener('input', autosize); autosize();

        // Upload
        const uploader = document.getElementById('uploader');
        uploader.addEventListener('submit', async (e)=>{
          e.preventDefault(); const f = document.getElementById('file'); if(!f.files.length){ showToast('Dosya seç!'); return; }
          const fd = new FormData(); fd.append('file', f.files[0]);
          const res = await fetch('/upload', { method:'POST', body: fd });
          const j = await res.json();
          docinfo.textContent = j.message || ('Yüklendi: ' + (f.files[0]?.name||''));
          showToast('Yüklendi');
        });

        function renderSources(sources){
          if(!useRag.checked || !sources || !sources.length){ return; }
          sourcesBox.innerHTML = '<div class="muted">Kaynaklar:</div>' + sources.map((s,i)=>`<div>${i+1}. <a data-id="${s.id}"><strong>${s.source}</strong></a> — <span class="muted">${(s.snippet||'').slice(0,160)}...</span></div>`).join('');
          sourcesBox.classList.remove('hidden');
          sourcesBox.querySelectorAll('a[data-id]').forEach(a=>{
            a.addEventListener('click', async ()=>{
              try{ const rid = a.getAttribute('data-id'); const r = await fetch('/source?id=' + encodeURIComponent(rid)); const j = await r.json(); mtext.textContent = j.text || ''; msource.textContent = j.source || ''; modal.style.display = 'flex'; }catch(e){ showToast('Kaynak açılamadı'); }
            });
          });
        }

        document.getElementById('close').addEventListener('click', ()=>{ modal.style.display='none'; mtext.textContent=''; msource.textContent=''; });
        document.getElementById('copySrc').addEventListener('click', ()=>{ navigator.clipboard?.writeText(mtext.textContent||''); showToast('Kaynak kopyalandı'); });
        modal.addEventListener('click', (e)=>{ if(e.target===modal){ modal.style.display='none'; } });

        let lastUser = '';
        async function send(textOpt=null, isRetry=false){
          const text = (textOpt ?? promptEl.value).trim(); if(!text) return;
          if(!isRetry){ promptEl.value=''; autosize(); append('user', text); } lastUser = text; setLoading(true);
          const body = { message: text, model: modelEl.value, temperature: parseFloat(tempEl.value) || 0.4, rag: useRag.checked };
          sourcesBox.classList.add('hidden'); sourcesBox.innerHTML='';
          try {
            if(useStream.checked){
              const res = await fetch('/chat_stream', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
              if(!res.ok) throw new Error('HTTP ' + res.status);
              const reader = res.body.getReader(); const decoder = new TextDecoder(); let full=''; const bubble = append('assistant', '');
              while(true){ const {done, value} = await reader.read(); if(done) break; const chunk = decoder.decode(value); full += chunk; bubble.querySelector('.content').innerHTML = md(full); chat.scrollTop = chat.scrollHeight; }
              const sres = await fetch('/last_sources', { method:'GET' }); const sj = await sres.json(); renderSources(sj.sources||[]);
            } else {
              const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
              if(!res.ok) throw new Error('HTTP ' + res.status);
              const j = await res.json(); append('assistant', j.reply); renderSources(j.sources||[]);
            }
          } catch(e){ append('assistant', 'Hata: ' + (e?.message || e)); }
          finally { setLoading(false); }
        }

        function exportChat(){
          const items = Array.from(document.querySelectorAll('.msg')).map(el=>({
            role: el.classList.contains('u') ? 'user' : 'assistant',
            text: el.querySelector('.content')?.innerText || '',
            time: el.querySelector('.time')?.textContent || ''
          }));
          const blob = new Blob([JSON.stringify(items, null, 2)], {type:'application/json'});
          const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'chat_export.json'; a.click();
        }

        sendBtn.addEventListener('click', ()=> send());
        clearBtn.addEventListener('click', resetSession);
        exportBtn.addEventListener('click', exportChat);
        promptEl.addEventListener('keydown', (e)=>{ if((e.ctrlKey || e.metaKey) && e.key === 'Enter'){ send(); } });
        scrollBtn.addEventListener('click', ()=> chat.scrollTop = chat.scrollHeight);
        chat.addEventListener('scroll', ()=>{ const nearBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 40; scrollBtn.style.display = nearBottom ? 'none' : 'inline-block'; });

        ensureSession();
      </script>
    </body>
    </html>
    """

    app_state = {"force_mock_env": bool(os.getenv("FORCE_MOCK", "").strip()), "last_sources": []}

    @app.post("/session")
    async def create_session(request: Request):
        sid = request.cookies.get("sid")
        if not sid:
            sid = str(uuid.uuid4())
            resp = JSONResponse({"session_id": sid})
            resp.set_cookie("sid", sid, httponly=True, samesite="lax")
            return resp
        return JSONResponse({"session_id": sid})

    @app.post("/reset")
    async def reset(request: Request):
        sid = request.cookies.get("sid")
        if sid and sid in _ENGINES:
            _ENGINES[sid].reset()
        return PlainTextResponse("OK")

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        name = file.filename or "doc"
        raw = await file.read()
        added = add_documents(name, raw, api_key=os.getenv("OPENAI_API_KEY"))
        return JSONResponse({"ok": True, "message": f"{name} eklendi, {added} parça."})

    @app.get("/last_sources")
    async def last_sources():
        return JSONResponse({"sources": app_state.get("last_sources", [])})

    @app.get("/source")
    async def get_source(id: str = Query(..., description="Chunk id")):
        doc = _DOC_INDEX.get(id)
        if not doc:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"id": doc.id, "source": doc.source, "text": doc.text})

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        sid = request.cookies.get("sid") or str(uuid.uuid4())
        html = HTMLResponse(INDEX_HTML)
        if not request.cookies.get("sid"):
            html.set_cookie("sid", sid, httponly=True, samesite="lax")
        return html

    def _build_context_and_sources(q: str, api_key: Optional[str], use_rag: bool) -> Tuple[str, List[Dict[str, str]]]:
        if not use_rag or not _DOCS:
            return "", []
        hits = retrieve(q, api_key=api_key, k=4)
        context = "\n\n".join([t for (_id, t, _s) in hits])
        sources = [{"id": _id, "source": s, "snippet": t[:240]} for (_id, t, s) in hits]
        return context, sources

    @app.post("/chat")
    async def chat(request: Request):
        data = await request.json()
        text = str(data.get("message", "")).strip()
        model = str(data.get("model", "gpt-4o-mini"))
        temperature = float(data.get("temperature", 0.4))
        use_rag = bool(data.get("rag", False))
        if not text:
            return JSONResponse({"reply": "", "sources": []})
        sid = request.cookies.get("sid") or str(uuid.uuid4())
        force_mock = app_state["force_mock_env"]
        eng = _get_engine_for_session(sid, force_mock=force_mock, model=model, temperature=temperature)
        ctx, sources = _build_context_and_sources(text, os.getenv("OPENAI_API_KEY"), use_rag)
        reply = eng.send(text, context=ctx)
        app_state["last_sources"] = sources
        resp = JSONResponse({"reply": reply, "session_id": sid, "sources": sources})
        if not request.cookies.get("sid"):
            resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp

    @app.post("/chat_stream")
    async def chat_stream(request: Request):
        data = await request.json()
        text = str(data.get("message", "")).strip()
        model = str(data.get("model", "gpt-4o-mini"))
        temperature = float(data.get("temperature", 0.4))
        use_rag = bool(data.get("rag", False))
        if not text:
            return PlainTextResponse("")
        sid = request.cookies.get("sid") or str(uuid.uuid4())
        force_mock = app_state["force_mock_env"]
        eng = _get_engine_for_session(sid, force_mock=force_mock, model=model, temperature=temperature)
        ctx, sources = _build_context_and_sources(text, os.getenv("OPENAI_API_KEY"), use_rag)

        def gen():
            for chunk in eng.stream(text, context=ctx):
                yield chunk
        app_state["last_sources"] = sources
        return StreamingResponse(gen(), media_type="text/plain")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# ------------------------------
# Tests (offline, fast)
# ------------------------------
import unittest

class TestChatEngine(unittest.TestCase):
    def test_mock_engine_echoes_user_intent(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        out = eng.send("Merhaba dünya")
        self.assertTrue(out.startswith("MOCK[succinct]:"))
        self.assertIn("Merhaba dünya", out)

    def test_history_accumulates(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        eng.send("Adımı hatırla: Ufuk")
        out2 = eng.send("Adım neydi?")
        self.assertIn("Adım neydi?", out2)

    def test_langchain_path_graceful_when_missing(self):
        eng = ChatEngine(model_name="gpt-4o-mini", temperature=0.2, api_key=None, force_mock=False)
        out = eng.send("Test")
        self.assertTrue(len(out) > 0)

    def test_langchain_init_with_api_key_graceful(self):
        eng = ChatEngine(model_name="gpt-4o-mini", temperature=0.2, api_key="sk-test-123", force_mock=False)
        out = eng.send("Test with key")
        self.assertTrue(len(out) > 0)

    def test_send_with_context(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        out = eng.send("Nedir?", context="Bu bir bağlamdır.")
        self.assertIn("MOCK[succinct]", out)


def _run_tests_and_exit() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestChatEngine)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ------------------------------
# Entrypoint
# ------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangChain Chatbot — CLI+API")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name when LangChain+OpenAI is available")
    p.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    p.add_argument("--mock", action="store_true", help="Force mock mode (no external calls)")
    p.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")
    p.add_argument("--api", action="store_true", help="Run FastAPI web server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.run_tests or os.getenv("RUN_TESTS") == "1":
        _run_tests_and_exit()
    if args.api:
        run_api(args)
    else:
        run_cli(args)
