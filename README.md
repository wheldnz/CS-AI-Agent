# AI Engineer Portfolio

Koleksi proyek AI yang mendemonstrasikan kemampuan membangun aplikasi AI production-ready, dari RAG pipeline sampai AI Agent dengan tool calling.

---

## Proyek

### 1. DocuChat AI -- RAG Document Q&A
Upload dokumen (PDF/TXT), tanya apa saja, dapatkan jawaban akurat beserta sumber referensi.

**Fitur utama:**
- Hybrid Search (BM25 keyword + Semantic vector search)
- Multi-Query Retrieval -- AI generate variasi pertanyaan untuk hasil pencarian lebih luas
- Multilingual Embeddings -- memahami Bahasa Indonesia dengan baik
- Conversation Memory -- ingat konteks percakapan
- Multi-Provider LLM (Groq/OpenAI/Gemini/OpenRouter)

```bash
streamlit run docuchat_ai.py
```

---

### 2. TeknoShop AI Agent -- Customer Service Agent
AI Agent yang bisa memanggil tools secara otomatis untuk membantu customer.

**Fitur utama:**
- Tool Calling / Function Calling -- agent memilih tool yang tepat
- ReAct Pattern -- multi-step reasoning (berpikir, bertindak, observasi)
- 5 Tools: cari produk, hitung diskon, cek pesanan, lihat katalog, cari di dokumen
- Custom Data -- upload CSV produk/pesanan sendiri
- RAG Document Search -- upload dokumen, agent bisa cari info di dalamnya
- Conversation Memory

```bash
streamlit run teknoshop_agent.py
```

---

## Arsitektur

### DocuChat AI (RAG Pipeline)
```
User bertanya
    |
    v
[Multi-Query] --> Generate 3 variasi pertanyaan
    |
    v
[Hybrid Search] --> BM25 (keyword) + Chroma (semantic)
    |
    v
[LLM] --> Susun jawaban + sumber referensi
    |
    v
Jawaban + Sources
```

### TeknoShop AI Agent (Tool Calling)
```
User bertanya
    |
    v
[AI Agent] --> Analisis pertanyaan, pilih tool yang tepat
    |
    +---> cari_produk()      --> cek stok & harga
    +---> hitung_diskon()    --> kalkulasi harga
    +---> cek_pesanan()      --> lacak pengiriman
    +---> lihat_katalog()    --> browsing produk
    +---> cari_di_dokumen()  --> RAG search di dokumen
    |
    v
[Compile results] --> Jawaban berdasarkan data asli
```

---

## Setup Lokal

### 1. Clone
```bash
git clone https://github.com/USERNAME/ai-engineer-portfolio.git
cd ai-engineer-portfolio
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Jalankan
```bash
# RAG App
streamlit run docuchat_ai.py

# Agent App
streamlit run teknoshop_agent.py
```

### 4. API Key
Dapatkan API key gratis dari salah satu provider:
- [Groq](https://console.groq.com) -- gratis, model Llama 3.3 70B
- [Google Gemini](https://ai.google.dev) -- gratis, model Gemini 2.0 Flash
- [OpenRouter](https://openrouter.ai) -- gratis, berbagai model
- [OpenAI](https://platform.openai.com) -- berbayar, GPT-4o

---

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| UI Framework | Streamlit |
| LLM Providers | Groq, OpenAI, Gemini, OpenRouter |
| Embeddings | HuggingFace (multilingual-MiniLM) |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| Document Loader | PyPDF, TextLoader |
| Orchestration | LangChain |

---

## Skill yang Didemonstrasikan

- **RAG** -- Retrieval Augmented Generation dengan hybrid search
- **AI Agent** -- Tool calling dengan ReAct pattern
- **Prompt Engineering** -- System prompt design untuk kontrol output
- **Multi-Provider** -- Abstraksi LLM supaya bisa ganti provider tanpa ubah kode
- **Error Handling** -- Fallback graceful saat tool calling gagal
- **Data Flexibility** -- Upload CSV custom, bukan hardcoded demo
