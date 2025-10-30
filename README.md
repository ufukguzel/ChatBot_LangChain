# LangChain Chatbot — CLI + Web UI (FastAPI) with Streaming & RAG

## 🚀 Özellikler

Bu proje, **LangChain** tabanlı bir chatbot uygulamasıdır. Hem **CLI (komut satırı)** hem de **FastAPI tabanlı web arayüzü** ile çalışır.

### 🔧 Desteklenen Özellikler
- **CLI Chatbot:** Harici bağımlılıklar olmadan çalışır (mock mod destekli).
- **FastAPI Web UI:** Modern arayüz, tema değiştirici, kopyalama, yenileme ve otomatik boyutlandırma özellikleri.
- **Gerçek Zamanlı Akış (Streaming):** Model yanıtlarını token bazında akışla gösterir.
- **Dosya Yükleme ve RAG:** TXT, MD ve **PDF** dosyalarıyla kaynak tabanlı yanıt üretimi.
- **Kaynak Görüntüleme:** Yanıtların altında tıklanabilir kaynak listesi ve modal pencere ile detay gösterimi.
- **Mock Modu:** API anahtarı veya bağımlılıklar eksikse otomatik olarak devreye girer.

---

## 📦 Kurulum

```bash
# Gerekli temel kütüphaneler
pip install fastapi uvicorn

# Opsiyonel (daha iyi RAG ve PDF desteği için)
pip install langchain-openai langchain-core langchain-text-splitters faiss-cpu python-dotenv pypdf
```

Ortam değişkenlerini ayarlayın:
```bash
# .env dosyasına ekleyin
OPENAI_API_KEY=sk-xxxxxx
```

---

## 🖥️ Kullanım

### 1. CLI Modu (Terminal Üzerinden)
```bash
python app.py              # Normal mod
python app.py --mock       # Mock modda çalıştır
python app.py --run-tests  # Testleri çalıştır
```

### 2. Web Arayüzü (FastAPI)
```bash
python app.py --api --host 0.0.0.0 --port 8000
# Ardından tarayıcıdan aç:
# http://localhost:8000
```

### 3. Çalışma Modları
| Mod | Açıklama |
|------|-----------|
| **Mock Mode** | LangChain veya API key yoksa otomatik çalışır |
| **LangChain Mode** | API key varsa LangChain/OpenAI entegre çalışır |
| **RAG Mode** | Yüklenen dosyalarla bağlam tabanlı yanıt üretir |

---

## 🧠 Özellik Detayları

### 🗂️ Dosya Yükleme (RAG)
- **Desteklenen formatlar:** `.txt`, `.md`, `.pdf`
- PDF’lerden metin çıkarımı için `pypdf` kullanılır.
- Metinler 700 karakterlik parçalara bölünür ve FAISS ile vektör arama yapılır.
- API anahtarı yoksa anahtar kelime tabanlı arama (fallback) kullanılır.

### 🔁 Streaming (Gerçek Zamanlı Yanıt)
- Yanıtlar parça parça (`chunk`) gelir.
- Akış bitince kaynaklar otomatik olarak görüntülenir.

### 💾 Mock Modu
- Dış bağımlılıklar olmadan test edilebilir.
- Deterministik yanıt üretir (aynı girişe aynı sonuç döner).

### 🧩 Arayüz Özellikleri
- Tema değiştirici (açık/koyu) — kalıcı ayar.
- “Kaynakları kullan” ve “Akışlı yanıt” seçenekleri.
- Dosya yükleme formu, tıklanabilir kaynak listesi.
- “Kopyala”, “Yenile” butonları ve JSON dışa aktarım.

---

## ✅ Testler

```bash
python app.py --run-tests
```

Tüm testler **çevrimdışı** ve **hızlı** çalışır. Mock modda yürütülür.

Test edilenler:
- Mock motorunun doğru yanıt üretimi
- Geçmiş (history) birikimi
- API key ile/olmayan senaryolar
- Bağlam kullanımı (context)

---

## 📁 Proje Yapısı

```
app.py               # Ana uygulama (CLI + Web API)
README.md            # Bu dosya
.env (opsiyonel)     # API anahtarı
```

---

## 💡 Gelecek Geliştirmeler

- ✅ Gerçek SSE tabanlı streaming (EventSource)
- 🔍 Kaynak modalında arama/filtreleme
- 📂 Çoklu dosya yükleme ve yönetim
- 💬 Kod bloklarına özel “kopyala” ikonu

---

## 🧑‍💻 Katkıda Bulunma

Pull request’ler memnuniyetle karşılanır. 🎉

---

**Hazırlayan:** Profesör 👨‍💻  
**Teknolojiler:** Python, FastAPI, LangChain, FAISS, HTML/CSS/JS
