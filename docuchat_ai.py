"""
DocuChat AI -- RAG Document Q&A
================================
Upload dokumen, tanya apa saja, dapatkan jawaban akurat beserta sumber referensi.

Teknik yang digunakan:
  - Hybrid Search (BM25 keyword + Semantic vector search)
  - Multi-Query Retrieval (generate variasi pertanyaan)
  - Multilingual Embeddings
  - Multi-provider LLM (Groq, OpenAI, Gemini, OpenRouter)
  - Conversation Memory

Jalankan:
  streamlit run docuchat_ai.py
"""

import streamlit as st
import tempfile
import os
import numpy as np
from pathlib import Path
from openai import OpenAI
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# ============================================================
# CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="page_facing_up",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6366F1, #8B5CF6, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1E1B4B, #312E81);
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #4338CA;
    }
    .stat-card h2 { margin: 0; color: #C7D2FE; font-size: 1.8rem; }
    .stat-card p { margin: 0.2rem 0 0; color: #818CF8; font-size: 0.85rem; }
    
    .source-card {
        background: #1E1B4B;
        border-left: 3px solid #818CF8;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.4rem 0;
        color: #E2E8F0 !important;
        font-size: 0.85rem;
    }
    .source-card strong { color: #A5B4FC !important; }
    .source-card em { color: #94A3B8 !important; }
    
    .technique-badge {
        display: inline-block;
        background: #312E81;
        color: #C7D2FE;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin: 0.1rem;
        border: 1px solid #4338CA;
    }
    
    .how-it-works {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .how-it-works h4 { color: #E2E8F0; margin-top: 0; }
    .how-it-works p { color: #94A3B8; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>DocuChat AI</h1>
    <p>Upload dokumen -- Tanya apa saja -- Jawaban akurat + sumber referensi</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LLM PROVIDERS
# ============================================================

PROVIDERS = {
    "Groq (Gratis)": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        "placeholder": "gsk_...",
        "signup": "https://console.groq.com",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "placeholder": "sk-...",
        "signup": "https://platform.openai.com",
    },
    "Google Gemini (Gratis)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-1.5-pro"],
        "placeholder": "AIza...",
        "signup": "https://ai.google.dev",
    },
    "OpenRouter (Gratis)": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-8b:free",
            "google/gemini-2.0-flash-exp:free",
        ],
        "placeholder": "sk-or-...",
        "signup": "https://openrouter.ai",
    },
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Pengaturan")
    
    provider = st.selectbox("AI Provider", list(PROVIDERS.keys()))
    pcfg = PROVIDERS[provider]
    
    api_key = st.text_input(
        f"API Key ({provider.split(' (')[0]})",
        type="password",
        placeholder=pcfg["placeholder"],
        help=f"Daftar di {pcfg['signup']}"
    )
    model_name = st.selectbox("Model", pcfg["models"])
    
    st.divider()
    st.subheader("Upload Dokumen")
    uploaded_files = st.file_uploader(
        "PDF atau TXT", type=["pdf", "txt"], accept_multiple_files=True
    )
    
    st.divider()
    st.subheader("Pengaturan Pencarian")
    
    search_mode = st.radio(
        "Mode Pencarian",
        ["Hybrid (Rekomendasi)", "Semantic saja", "Keyword saja"],
        help="Hybrid menggabungkan keyword + semantic search untuk hasil terbaik"
    )
    
    use_multi_query = st.checkbox(
        "Multi-Query Retrieval",
        value=True,
        help="AI generate variasi pertanyaan supaya pencarian lebih luas"
    )
    
    with st.expander("Pengaturan Lanjutan"):
        chunk_size = st.slider("Chunk Size", 300, 2000, 800, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        top_k = st.slider("Jumlah Referensi (k)", 3, 15, 6)
        bm25_weight = st.slider(
            "Bobot Keyword vs Semantic", 0.0, 1.0, 0.4, 0.1,
            help="0 = full semantic, 1 = full keyword, 0.4 = campuran (rekomendasi)"
        )
    
    st.divider()
    show_debug = st.checkbox("Tampilkan Debug", value=False)
    
    if st.button("Reset Semua", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ============================================================
# RAG CORE FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load model embedding multilingual (di-cache supaya tidak download ulang)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def process_documents(files, chunk_size, chunk_overlap):
    """
    Proses dokumen dari file mentah sampai siap dicari.
    
    Alur:
    1. Baca file (PDF/TXT) -> extract teks
    2. Pecah teks jadi chunks
    3. Buat vector index (Chroma) untuk semantic search
    4. Buat BM25 index untuk keyword search
    """
    all_docs = []
    extracted_text = ""
    file_names = []
    
    for file in files:
        file_names.append(file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
                extracted_text += f"\n--- {file.name} (hal. {doc.metadata.get('page', '?')}) ---\n"
                extracted_text += doc.page_content + "\n"
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    chunks = splitter.split_documents(all_docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    
    if not chunks:
        raise ValueError("Tidak ada teks yang berhasil di-extract dari dokumen.")
    
    # Buat vector index (semantic search)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    # Buat BM25 index (keyword search)
    chunk_texts = [c.page_content for c in chunks]
    tokenized = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized)
    
    stats = {
        "files": file_names,
        "total_pages": len(all_docs),
        "total_chunks": len(chunks),
    }
    
    return vectorstore, bm25, chunk_texts, chunks, stats, extracted_text


def generate_query_variations(question, api_key, base_url, model):
    """
    Multi-Query Retrieval: generate variasi pertanyaan.
    
    Kenapa perlu? User mungkin bertanya "boleh WFH gak?" tapi di dokumen
    tertulis "Kebijakan kerja dari rumah". Variasi pertanyaan membantu
    menemukan chunk yang relevan meskipun kata-katanya berbeda.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"""Berikan 3 variasi berbeda dari pertanyaan berikut.
Setiap variasi harus menggunakan kata/istilah yang berbeda tapi maknanya sama.
Tulis satu pertanyaan per baris, TANPA nomor atau bullet.

Pertanyaan: {question}"""
        }],
        temperature=0.7,
        max_tokens=300,
    )
    
    raw = response.choices[0].message.content
    variations = [q.strip() for q in raw.strip().split("\n") if q.strip() and len(q.strip()) > 5]
    return variations[:3]


def hybrid_search(query, vectorstore, bm25, chunk_texts, chunks, top_k, alpha):
    """
    Hybrid Search: gabungkan keyword (BM25) + semantic search.
    
    alpha: bobot keyword vs semantic
        0.0 = full semantic
        0.5 = seimbang
        1.0 = full keyword
    
    Cara kerja:
    1. Jalankan BM25 search -> skor keyword per chunk
    2. Jalankan semantic search -> skor semantic per chunk
    3. Normalisasi kedua skor ke range 0-1
    4. Gabungkan: final = alpha * bm25 + (1-alpha) * semantic
    5. Sort, ambil top-k
    """
    scores = {}
    
    # BM25 (keyword) scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    
    for i, score in enumerate(bm25_scores):
        scores[i] = {"bm25": score / max_bm25, "semantic": 0}
    
    # Semantic scores
    semantic_results = vectorstore.similarity_search_with_score(
        query, k=min(top_k * 2, len(chunks))
    )
    
    for doc, distance in semantic_results:
        similarity = 1 / (1 + distance)
        for i, ct in enumerate(chunk_texts):
            if ct == doc.page_content:
                scores.setdefault(i, {"bm25": 0, "semantic": 0})
                scores[i]["semantic"] = similarity
                break
    
    # Gabungkan dengan bobot
    final_scores = []
    for i, s in scores.items():
        combined = alpha * s["bm25"] + (1 - alpha) * s["semantic"]
        if combined > 0:
            final_scores.append((i, combined))
    
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return [(chunks[idx], score) for idx, score in final_scores[:top_k]]


def search_documents(query, vectorstore, bm25, chunk_texts, chunks,
                     top_k, search_mode, alpha):
    """Wrapper: pilih metode search sesuai pilihan user."""
    if search_mode == "Hybrid (Rekomendasi)":
        return hybrid_search(query, vectorstore, bm25, chunk_texts, chunks, top_k, alpha)
    elif search_mode == "Keyword saja":
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_idx = np.argsort(bm25_scores)[::-1][:top_k]
        return [(chunks[i], bm25_scores[i]) for i in top_idx if bm25_scores[i] > 0]
    else:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        return [(doc, 1/(1+score)) for doc, score in results]


# ============================================================
# MAIN APP
# ============================================================

if not api_key:
    st.info("Masukkan API Key di sidebar untuk mulai.")
    st.stop()

if not uploaded_files:
    # Landing page
    st.markdown("""
    <div class="how-it-works">
        <h4>Cara Menggunakan</h4>
        <p>1. Upload dokumen PDF atau TXT di sidebar</p>
        <p>2. Ketik pertanyaan di chat box</p>
        <p>3. AI menjawab berdasarkan isi dokumen + menunjukkan sumber</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Teknik AI yang digunakan:**")
    techniques = [
        "Hybrid Search (BM25 + Semantic)",
        "Multi-Query Retrieval",
        "Multilingual Embeddings",
        "Multi-Provider LLM",
        "Conversation Memory",
    ]
    cols = st.columns(len(techniques))
    for col, tech in zip(cols, techniques):
        col.markdown(f'<span class="technique-badge">{tech}</span>', unsafe_allow_html=True)
    
    st.stop()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

file_names = sorted([f.name for f in uploaded_files])
if "processed_files" not in st.session_state or st.session_state.processed_files != file_names:
    with st.spinner("Memproses dokumen (split, embed, indexing)..."):
        vectorstore, bm25, chunk_texts, chunks, stats, extracted_text = process_documents(
            uploaded_files, chunk_size, chunk_overlap
        )
        st.session_state.vectorstore = vectorstore
        st.session_state.bm25 = bm25
        st.session_state.chunk_texts = chunk_texts
        st.session_state.chunks = chunks
        st.session_state.doc_stats = stats
        st.session_state.processed_files = file_names
        st.session_state.extracted_text = extracted_text
        st.session_state.messages = []
    st.success("Dokumen berhasil diproses!")

stats = st.session_state.doc_stats

# Stats row
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="stat-card"><h2>{len(stats["files"])}</h2><p>Dokumen</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card"><h2>{stats["total_pages"]}</h2><p>Halaman</p></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card"><h2>{stats["total_chunks"]}</h2><p>Chunks</p></div>', unsafe_allow_html=True)
mode_label = search_mode.split(" ")[0]
c4.markdown(f'<div class="stat-card"><h2>{mode_label}</h2><p>Search Mode</p></div>', unsafe_allow_html=True)

# Debug panel (opsional)
if show_debug:
    with st.expander("DEBUG: Extracted Text"):
        st.text_area("Raw Text", st.session_state.get("extracted_text", ""), height=200)
    with st.expander("DEBUG: Chunks"):
        for i, c in enumerate(st.session_state.get("chunks", [])[:10]):
            st.code(c.page_content[:200], language=None)

st.divider()


# ============================================================
# CHAT INTERFACE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Tanya sesuatu tentang dokumen..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Mencari & menyusun jawaban..."):
            try:
                all_queries = [user_input]
                
                # Multi-Query: generate variasi
                if use_multi_query:
                    try:
                        variations = generate_query_variations(
                            user_input, api_key, pcfg["base_url"], model_name
                        )
                        all_queries.extend(variations)
                        if show_debug:
                            with st.expander("DEBUG: Query Variations"):
                                for v in variations:
                                    st.write(f"- {v}")
                    except Exception:
                        pass  # Tetap lanjut dengan single query
                
                # Search untuk setiap query, gabungkan hasil
                all_results = {}
                for q in all_queries:
                    results = search_documents(
                        q,
                        st.session_state.vectorstore,
                        st.session_state.bm25,
                        st.session_state.chunk_texts,
                        st.session_state.chunks,
                        top_k, search_mode, bm25_weight
                    )
                    for doc, score in results:
                        key = hash(doc.page_content)
                        if key not in all_results or all_results[key][1] < score:
                            all_results[key] = (doc, score)
                
                sorted_results = sorted(
                    all_results.values(), key=lambda x: x[1], reverse=True
                )[:top_k]
                
                if show_debug:
                    with st.expander(f"DEBUG: {len(sorted_results)} chunks ditemukan"):
                        for doc, score in sorted_results:
                            st.markdown(f"**Score: {score:.3f}** | {doc.metadata.get('source', '?')}")
                            st.code(doc.page_content[:200], language=None)
                
                # Susun konteks dari chunk yang ditemukan
                context_parts = []
                for doc, _ in sorted_results:
                    src = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "?")
                    context_parts.append(f"[Sumber: {src}, Halaman: {page}]\n{doc.page_content}")
                context = "\n\n===\n\n".join(context_parts)
                
                # Riwayat percakapan
                history_text = ""
                recent_msgs = st.session_state.messages[-6:]
                for msg in recent_msgs[:-1]:
                    role = "User" if msg["role"] == "user" else "Asisten"
                    history_text += f"{role}: {msg['content']}\n"
                
                # Generate jawaban dari LLM
                llm_client = OpenAI(api_key=api_key, base_url=pcfg["base_url"])
                
                system_msg = """Kamu adalah asisten AI yang menjawab pertanyaan berdasarkan dokumen.
ATURAN:
- Jawab berdasarkan konteks dokumen yang diberikan
- Jika ada informasi relevan, jawab dengan lengkap
- Kutip bagian dokumen yang mendukung jawaban
- Jangan mengarang informasi yang tidak ada di dokumen
- Jika konteks tidak cukup, bilang dengan jujur
- Jawab dalam Bahasa Indonesia"""

                user_msg = f"""Konteks dokumen:
{context}

Riwayat percakapan:
{history_text}

Pertanyaan: {user_input}

Jawab berdasarkan konteks dokumen di atas:"""

                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                answer = response.choices[0].message.content
                
                st.markdown(answer)
                
                # Tampilkan sumber referensi
                if sorted_results:
                    sources_html = []
                    for doc, score in sorted_results[:5]:
                        src = doc.metadata.get("source", "?")
                        page = doc.metadata.get("page", "?")
                        preview = doc.page_content[:100].replace("\n", " ")
                        sources_html.append(
                            f'<div class="source-card">'
                            f'<strong>{src}</strong> (hal. {page}) '
                            f'| relevansi: {score:.2f}<br>'
                            f'<em>{preview}...</em></div>'
                        )
                    
                    with st.expander(f"Sumber Referensi ({len(sorted_results)} dokumen)"):
                        st.markdown("".join(sources_html), unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", "content": answer
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if show_debug:
                    import traceback
                    st.code(traceback.format_exc(), language=None)
