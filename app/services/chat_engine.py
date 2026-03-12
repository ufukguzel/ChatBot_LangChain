"""
Chat engine service.

Provides:
  - MockLLM          — deterministic offline LLM, used when no API key is present
  - SimpleHistory    — lightweight framework-agnostic history store
  - ChatEngine       — unified interface over LangChain or MockLLM
  - ChatEngineRegistry — thread-safe per-session engine registry

Graceful degradation:
  1. LangChain packages missing           → MockLLM
  2. OPENAI_API_KEY missing               → MockLLM
  3. force_mock=True                      → MockLLM
  4. LangChain setup fails at runtime     → falls back to MockLLM
  5. LangChain API call fails at runtime  → returns "[Hata: ...]" string, never raises
"""
import logging
import threading
import uuid
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LangChain imports
# ---------------------------------------------------------------------------

_LC_AVAILABLE = True
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
    from langchain_core.runnables.history import RunnableWithMessageHistory  # type: ignore
    from langchain_core.chat_history import InMemoryChatMessageHistory  # type: ignore
except ImportError:
    _LC_AVAILABLE = False
    logger.info("LangChain packages not found — mock mode will be used.")


# ---------------------------------------------------------------------------
# Mock / fallback implementations
# ---------------------------------------------------------------------------

class SimpleHistory:
    """Minimal chat history — framework-agnostic."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_ai(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def as_text(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)


class MockLLM:
    """
    Deterministic mock that echoes the latest user message.
    Keeps all tests offline and fully predictable.
    """

    def invoke(self, prompt_text: str) -> str:
        # Extract the last "user:" line from history text
        lines = [l for l in prompt_text.strip().splitlines() if l.startswith("user:")]
        last = lines[-1].replace("user:", "").strip() if lines else prompt_text.strip()
        return f"MOCK[succinct]: {last}"

    def stream(self, text: str) -> Generator[str, None, None]:
        for token in text.split():
            yield token + " "


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Sen deneyimli bir yardımcı asistansın. Yanıtlarında kısa, net ve doğru ol. "
    "Kullanıcı Türkçe konuşuyorsa Türkçe yanıt ver. Gerekirse adım adım açıkla."
)


class ChatEngine:
    """
    Unified LLM interface.  Instantiate once per session.

    Parameters
    ----------
    model_name:  OpenAI model identifier (ignored in mock mode)
    temperature: Sampling temperature
    api_key:     OpenAI API key; None → mock mode
    session_id:  Stable ID for history lookup; generated if omitted
    force_mock:  True → always use MockLLM regardless of api_key
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str],
        session_id: Optional[str] = None,
        force_mock: bool = False,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

        self._use_langchain: bool = _LC_AVAILABLE and (not force_mock) and bool(api_key)
        self.history_map: Dict[str, object] = {}
        self._histories: Dict[str, SimpleHistory] = {}
        self._chain = None

        if self._use_langchain:
            try:
                self._setup_langchain(model_name, temperature, api_key)
            except Exception as exc:
                logger.warning(
                    "LangChain setup failed (%s) — falling back to mock mode.", exc
                )
                self._use_langchain = False

        if not self._use_langchain:
            self._mock = MockLLM()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _setup_langchain(self, model_name: str, temperature: float, api_key: str) -> None:
        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("system", "Bağlam (varsa):\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        base_chain = prompt | llm | StrOutputParser()

        def _get_history(sid: str) -> "InMemoryChatMessageHistory":
            if sid not in self.history_map:
                self.history_map[sid] = InMemoryChatMessageHistory()
            return self.history_map[sid]  # type: ignore[return-value]

        self._chain = RunnableWithMessageHistory(
            base_chain,
            _get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_simple_history(self, sid: str) -> SimpleHistory:
        if sid not in self._histories:
            self._histories[sid] = SimpleHistory()
        return self._histories[sid]

    def _lc_config(self) -> dict:
        return {"configurable": {"session_id": self.session_id}}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear history for this engine's session."""
        self._histories.pop(self.session_id, None)
        self.history_map.pop(self.session_id, None)

    def send(self, user_input: str, context: str = "") -> str:
        """Return a complete response string."""
        if self._use_langchain and self._chain is not None:
            try:
                result = self._chain.invoke(
                    {"input": user_input, "context": context},
                    config=self._lc_config(),
                )
                return str(result)
            except Exception as exc:
                logger.error("LangChain invoke error: %s", exc, exc_info=True)
                return f"[Hata: {exc}]"

        # Mock path — update history manually
        h = self._get_simple_history(self.session_id)
        h.add_user(user_input)
        prompt = h.as_text()
        if context:
            prompt += f"\nCTX: {context}"
        reply = self._mock.invoke(prompt)
        h.add_ai(reply)
        return reply

    def stream(self, user_input: str, context: str = "") -> Generator[str, None, None]:
        """Yield response tokens one by one."""
        if self._use_langchain and self._chain is not None:
            try:
                for chunk in self._chain.stream(
                    {"input": user_input, "context": context},
                    config=self._lc_config(),
                ):
                    yield str(chunk)
                return
            except Exception as exc:
                logger.error("LangChain stream error: %s", exc, exc_info=True)
                yield f"[Hata: {exc}]"
                return

        # Mock path
        for token in self._mock.stream(f"MOCK[succinct]: {user_input}"):
            yield token


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ChatEngineRegistry:
    """
    Thread-safe per-session engine store.

    One engine per session_id is created on first use.  Engines are cheap
    (no network call at construction time), so we don't need eviction for
    typical workloads; add LRU eviction here if memory becomes a concern.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engines: Dict[str, ChatEngine] = {}

    def get_or_create(
        self,
        session_id: str,
        *,
        force_mock: bool,
        model: str,
        temperature: float,
        api_key: Optional[str],
    ) -> ChatEngine:
        with self._lock:
            if session_id not in self._engines:
                self._engines[session_id] = ChatEngine(
                    model_name=model,
                    temperature=temperature,
                    api_key=api_key,
                    session_id=session_id,
                    force_mock=force_mock or (api_key is None),
                )
            return self._engines[session_id]

    def reset(self, session_id: str) -> None:
        """Reset history for *session_id* without destroying the engine."""
        with self._lock:
            engine = self._engines.get(session_id)
        if engine:
            engine.reset()

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._engines.pop(session_id, None)

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._engines)
