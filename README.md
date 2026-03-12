# LangChain Chatbot — Production-Ready

CLI + FastAPI chatbot with streaming responses, RAG (TXT/MD/PDF), clickable sources, and graceful degradation to offline mock mode.

---

## Özellikler

| Özellik | Açıklama |
|---------|----------|
| **CLI Chatbot** | Harici bağımlılık olmadan çalışır, mock modu destekli |
| **FastAPI Web UI** | Modern arayüz, tema değiştirici, kopyalama, yenileme |
| **Streaming** | Token bazlı gerçek zamanlı yanıt akışı |
| **RAG** | TXT / MD / PDF yükleme, FAISS vektör araması |
| **Kaynak Modal** | Tıklanabilir kaynak listesi ve detay penceresi |
| **Mock Modu** | API key veya LangChain yoksa otomatik devreye girer |

---

## Kurulum

```bash
# Python 3.10+ gereklidir
pip install -r requirements.txt
```

### Ortam Değişkenleri

```bash
cp .env.example .env
# .env dosyasını açıp OPENAI_API_KEY değerini girin
```

---

## Çalıştırma

### CLI (Komut Satırı)

```bash
python cli.py                         # Normal mod
python cli.py --mock                  # Zorunlu mock modu
python cli.py --model gpt-4o --temperature 0.7
```

### Web Sunucusu (FastAPI)

```bash
# Geliştirme (hot-reload ile)
python server.py --reload

# Production
python server.py --host 0.0.0.0 --port 8000

# Doğrudan uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Tarayıcıda: `http://localhost:8000`

### Geriye Dönük Uyumluluk (eski `app.py`)

```bash
python app.py                         # CLI
python app.py --mock                  # Mock CLI
python app.py --api --port 8000       # Web sunucu
python app.py --run-tests             # Testler
```

---

## Testler

```bash
# Tüm testler (offline, pytest)
pytest tests/ -v

# Belirli test dosyası
pytest tests/test_chat_engine.py -v
pytest tests/test_rag_service.py -v
pytest tests/test_api.py -v
```

Tüm testler çevrimdışı çalışır — gerçek OpenAI çağrısı yapılmaz.

---

## Proje Yapısı

```
ChatBot_LangChain/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI uygulama fabrikası
│   ├── api/
│   │   └── routes.py              # Tüm HTTP endpoint’leri
│   ├── core/
│   │   ├── config.py              # Ayarlar (.env okuma)
│   │   └── logging_setup.py       # Merkezi loglama yapılandırması
│   ├── models/
│   │   └── schemas.py             # Pydantic istek/yanıt modelleri
│   ├── services/
│   │   ├── chat_engine.py         # ChatEngine + MockLLM + Registry
│   │   ├── rag_service.py         # RAG servisi (FAISS + keyword fallback)
│   │   └── document_loader.py     # PDF + metin yükleme
│   └── templates/
│       └── index.html             # Web arayüzü (JS hataları giderildi)
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── test_chat_engine.py
│   ├── test_rag_service.py
│   ├── test_document_loader.py
│   └── test_api.py
├── cli.py                         # CLI giriş noktası
├── server.py                      # Web sunucu giriş noktası
├── app.py                         # Geriye dönük uyumluluk shim’i
├── requirements.txt
├── .env.example
└── README.md
```

---

## Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `OPENAI_API_KEY` | _(boş)_ | OpenAI API anahtarı; yoksa mock moda geçilir |
| `MODEL_NAME` | `gpt-4o-mini` | Kullanılacak LLM modeli |
| `TEMPERATURE` | `0.4` | Örnekleme sıcaklığı (0.0–2.0) |
| `FORCE_MOCK` | `0` | `1` = her zaman mock modu kullan |
| `HOST` | `127.0.0.1` | Sunucu bind adresi |
| `PORT` | `8000` | Sunucu portu |
| `ALLOWED_ORIGINS` | `http://localhost:8000,...` | CORS izin verilen kaynaklar |
| `MAX_UPLOAD_MB` | `10` | Maksimum dosya yükleme boyutu (MB) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Düzeltilen Hatalar (v1 → v2)

1. **JS `const uploader` çift tanımlama** — `SyntaxError` düzeltildi
2. **`/chat_stream` cookie set edilmiyordu** — `StreamingResponse`’a cookie eklendi
3. **`test_langchain_init_with_api_key_graceful`** — gerçek API çağrısı yapılmayacak şekilde düzeltildi
4. **Global mutable RAG state** — `RAGService` sınıfına taşındı, `threading.Lock` ile koruma altına alındı
5. **`app_state[“last_sources”]` race condition** — kaynaklar artık stream’in sonuna `\n__SOURCES__:{json}` olarak ekleniyor
6. **Mock `send()` çift history ekleme** — `h.as_text() + “\nuser: ...”` bug’ı giderildi
7. **CORS `allow_origins=[“*”]`** — env var ile yapılandırılabilir hale getirildi
8. **Upload validation yok** — dosya tipi ve boyut kontrolü eklendi
9. **Synchronous `gen()` in async handler** — async generator’a çevrildi
10. **`_ENGINES` dict temizlenmiyordu** — `ChatEngineRegistry` yapısına taşındı

---

## Mimari Kararlar

- **`RAGService`**: Tüm RAG state’i sınıf içine alındı, global değişken yok
- **`ChatEngineRegistry`**: Thread-safe session yönetimi, `threading.Lock` ile
- **Streaming sources**: `/last_sources` endpoint yerine stream sonu sentinel `\n__SOURCES__:` kullanıldı — race condition ortadan kalktı
- **Fallback zinciri**: LangChain kurulu değil → Mock | API key yok → Mock | LangChain setup hatası → Mock | API çağrısı hatası → `[Hata: ...]` string döner

---

## Bağımlılıklar

Çekirdek (web arayüzü için):
```
fastapi, uvicorn, python-multipart
```

LangChain (gerçek LLM için):
```
langchain-openai, langchain-core, langchain-text-splitters
```

Opsiyonel (RAG + PDF):
```
langchain-community, faiss-cpu, pypdf
```

Config + Test:
```
python-dotenv, pydantic, pytest, httpx
```

---

**Teknolojiler:** Python, FastAPI, LangChain, FAISS, HTML/CSS/JS
