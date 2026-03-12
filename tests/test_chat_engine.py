"""Unit tests for chat_engine.py — all offline."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.chat_engine import (
    ChatEngine,
    ChatEngineRegistry,
    MockLLM,
    SimpleHistory,
    _LC_AVAILABLE,
)


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

class TestMockLLM:
    def test_invoke_returns_user_text(self):
        llm = MockLLM()
        result = llm.invoke("user: hello world")
        assert "MOCK[succinct]" in result
        assert "hello world" in result

    def test_invoke_with_no_user_prefix(self):
        llm = MockLLM()
        result = llm.invoke("some prompt without user prefix")
        assert isinstance(result, str) and len(result) > 0

    def test_stream_yields_multiple_tokens(self):
        llm = MockLLM()
        tokens = list(llm.stream("hello world test"))
        assert len(tokens) == 3  # three words
        assert "".join(tokens).strip() == "hello world test"

    def test_stream_with_single_word(self):
        llm = MockLLM()
        tokens = list(llm.stream("hello"))
        assert len(tokens) == 1


# ---------------------------------------------------------------------------
# SimpleHistory
# ---------------------------------------------------------------------------

class TestSimpleHistory:
    def test_add_user_and_ai(self):
        h = SimpleHistory()
        h.add_user("hi")
        h.add_ai("hello")
        text = h.as_text()
        assert "user: hi" in text
        assert "assistant: hello" in text

    def test_empty_history(self):
        h = SimpleHistory()
        assert h.as_text() == ""

    def test_message_order_preserved(self):
        h = SimpleHistory()
        h.add_user("first")
        h.add_ai("second")
        h.add_user("third")
        msgs = [m["content"] for m in h.messages]
        assert msgs == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# ChatEngine — mock mode (always offline)
# ---------------------------------------------------------------------------

class TestChatEngineMockMode:
    def test_mock_engine_echoes_user_intent(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        out = eng.send("Merhaba dünya")
        assert out.startswith("MOCK[succinct]:")
        assert "Merhaba dünya" in out

    def test_history_accumulates(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        eng.send("Adımı hatırla: Ufuk")
        out2 = eng.send("Adım neydi?")
        # The mock echoes the latest user input from history
        assert "Adım neydi?" in out2

    def test_send_with_context_injected(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        out = eng.send("Nedir?", context="Bu bir bağlamdır.")
        assert "MOCK[succinct]" in out

    def test_stream_yields_chunks(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        chunks = list(eng.stream("test message"))
        assert len(chunks) > 0
        full = "".join(chunks)
        assert "test" in full

    def test_stream_is_generator(self):
        import types
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        gen = eng.stream("hi")
        assert isinstance(gen, types.GeneratorType)

    def test_reset_clears_history(self):
        eng = ChatEngine(model_name="mock", temperature=0.0, api_key=None, force_mock=True)
        eng.send("Hello")
        assert eng.session_id in eng._histories
        eng.reset()
        assert eng.session_id not in eng._histories

    def test_custom_session_id(self):
        eng = ChatEngine(
            model_name="mock", temperature=0.0, api_key=None,
            session_id="my-session", force_mock=True,
        )
        assert eng.session_id == "my-session"

    def test_no_api_key_forces_mock(self):
        eng = ChatEngine(model_name="gpt-4o-mini", temperature=0.2, api_key=None, force_mock=False)
        assert not eng._use_langchain
        out = eng.send("Test")
        assert len(out) > 0

    def test_force_mock_overrides_key(self):
        eng = ChatEngine(
            model_name="gpt-4o-mini", temperature=0.2,
            api_key="sk-real-would-be-here", force_mock=True,
        )
        assert not eng._use_langchain


# ---------------------------------------------------------------------------
# ChatEngine — LangChain path (mocked, offline)
# ---------------------------------------------------------------------------

class TestChatEngineLangChainPath:
    def test_langchain_path_uses_chain(self, mock_lc_chain):
        eng = ChatEngine(
            model_name="gpt-4o-mini", temperature=0.2,
            api_key="sk-test-key", force_mock=False,
        )
        if not eng._use_langchain:
            pytest.skip("LangChain not installed")
        out = eng.send("Hello LangChain")
        assert len(out) > 0
        mock_lc_chain.invoke.assert_called_once()

    def test_langchain_stream_uses_chain(self, mock_lc_chain):
        eng = ChatEngine(
            model_name="gpt-4o-mini", temperature=0.2,
            api_key="sk-test-key", force_mock=False,
        )
        if not eng._use_langchain:
            pytest.skip("LangChain not installed")
        chunks = list(eng.stream("Hello stream"))
        assert len(chunks) > 0

    def test_graceful_degradation_fake_key(self):
        """
        With a fake key and LangChain available, either:
          (a) Setup fails → falls back to mock → returns MOCK[...] string
          (b) Setup succeeds but invoke fails → returns [Hata: ...] string
        Either way, the output must be non-empty and no exception must propagate.
        """
        eng = ChatEngine(
            model_name="gpt-4o-mini", temperature=0.2,
            api_key="sk-test-invalid-key", force_mock=False,
        )
        out = eng.send("Test graceful degradation")
        assert isinstance(out, str) and len(out) > 0

    @patch("app.services.chat_engine._LC_AVAILABLE", False)
    def test_no_langchain_packages_fallback(self):
        eng = ChatEngine(
            model_name="gpt-4o-mini", temperature=0.2,
            api_key="sk-test-key", force_mock=False,
        )
        assert not eng._use_langchain
        out = eng.send("Test no LC")
        assert "MOCK[succinct]" in out


# ---------------------------------------------------------------------------
# ChatEngineRegistry
# ---------------------------------------------------------------------------

class TestChatEngineRegistry:
    def test_get_or_create_returns_same_engine(self):
        reg = ChatEngineRegistry()
        e1 = reg.get_or_create("s1", force_mock=True, model="m", temperature=0.4, api_key=None)
        e2 = reg.get_or_create("s1", force_mock=True, model="m", temperature=0.4, api_key=None)
        assert e1 is e2

    def test_different_sessions_get_different_engines(self):
        reg = ChatEngineRegistry()
        e1 = reg.get_or_create("s1", force_mock=True, model="m", temperature=0.4, api_key=None)
        e2 = reg.get_or_create("s2", force_mock=True, model="m", temperature=0.4, api_key=None)
        assert e1 is not e2

    def test_reset_does_not_raise(self):
        reg = ChatEngineRegistry()
        engine = reg.get_or_create("s3", force_mock=True, model="m", temperature=0.4, api_key=None)
        engine.send("hello")
        reg.reset("s3")   # should not raise

    def test_reset_nonexistent_session(self):
        reg = ChatEngineRegistry()
        reg.reset("does-not-exist")  # should not raise

    def test_remove_session(self):
        reg = ChatEngineRegistry()
        reg.get_or_create("s4", force_mock=True, model="m", temperature=0.4, api_key=None)
        assert reg.active_sessions() == 1
        reg.remove("s4")
        assert reg.active_sessions() == 0

    def test_active_sessions_count(self):
        reg = ChatEngineRegistry()
        for i in range(5):
            reg.get_or_create(f"sess-{i}", force_mock=True, model="m", temperature=0.4, api_key=None)
        assert reg.active_sessions() == 5
