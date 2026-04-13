"""
TeknoShop AI Agent -- Customer Service Agent
=============================================
AI Agent yang bisa menggunakan tools untuk membantu customer:
  - Cari produk dan cek stok
  - Hitung diskon dan simulasi harga
  - Lacak status pesanan
  - Cari informasi di dokumen yang diupload
  - Data produk/pesanan bisa di-custom lewat CSV

Teknik:
  - Tool Calling / Function Calling
  - ReAct Pattern (Reasoning + Acting)
  - RAG Document Search
  - Conversation Memory

Jalankan:
  streamlit run teknoshop_agent.py
"""

import streamlit as st
import json
import csv
import io
import os
import tempfile
from pathlib import Path
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# ============================================================
# CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="TeknoShop AI Agent",
    page_icon="page_facing_up",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.5rem;
    }
    .app-header h1 {
        background: linear-gradient(135deg, #10B981, #059669, #047857);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }
    
    .tool-log {
        background: #0C1222;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #94A3B8;
    }
    .tool-log .tool-name { color: #34D399; font-weight: 600; }
    .tool-log .tool-args { color: #818CF8; }
    .tool-log .tool-result { color: #FBBF24; }
    
    .product-card {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .product-card h4 { color: #E2E8F0; margin: 0 0 0.3rem; }
    .product-card p { color: #94A3B8; margin: 0.2rem 0; font-size: 0.9rem; }
    .product-card .price { color: #34D399; font-weight: 700; font-size: 1.1rem; }
    
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #065F46; color: #6EE7B7; }
    .badge-red { background: #7F1D1D; color: #FCA5A5; }
    .badge-blue { background: #1E3A5F; color: #93C5FD; }
    
    .stat-box {
        background: linear-gradient(135deg, #064E3B, #065F46);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #10B981;
    }
    .stat-box h3 { margin: 0; color: #A7F3D0; font-size: 1.6rem; }
    .stat-box p { margin: 0.2rem 0 0; color: #6EE7B7; font-size: 0.8rem; }
    
    .reasoning-step {
        background: #0F172A;
        border-left: 3px solid #10B981;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #CBD5E1;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>TeknoShop AI Agent</h1>
    <p>Customer Service Agent -- Cek Produk, Hitung Diskon, Lacak Pesanan</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DEFAULT DATA
# ============================================================

DEFAULT_PRODUCTS = {
    "iphone 15": {
        "nama": "iPhone 15", "harga": 14999000, "stok": 23,
        "warna": ["Hitam", "Putih", "Biru", "Pink"], "kategori": "Smartphone",
    },
    "iphone 15 pro": {
        "nama": "iPhone 15 Pro", "harga": 19999000, "stok": 12,
        "warna": ["Titanium Natural", "Titanium Biru", "Titanium Hitam"], "kategori": "Smartphone",
    },
    "samsung s24": {
        "nama": "Samsung Galaxy S24", "harga": 11999000, "stok": 15,
        "warna": ["Hitam", "Hijau", "Ungu"], "kategori": "Smartphone",
    },
    "samsung s24 ultra": {
        "nama": "Samsung Galaxy S24 Ultra", "harga": 19499000, "stok": 8,
        "warna": ["Titanium Gray", "Titanium Violet"], "kategori": "Smartphone",
    },
    "pixel 8": {
        "nama": "Google Pixel 8", "harga": 9999000, "stok": 0,
        "warna": [], "kategori": "Smartphone",
    },
    "macbook air m3": {
        "nama": "MacBook Air M3", "harga": 17999000, "stok": 5,
        "warna": ["Silver", "Space Gray", "Midnight"], "kategori": "Laptop",
    },
    "ipad air": {
        "nama": "iPad Air M2", "harga": 10499000, "stok": 18,
        "warna": ["Space Gray", "Starlight", "Purple", "Blue"], "kategori": "Tablet",
    },
    "airpods pro": {
        "nama": "AirPods Pro 2", "harga": 3799000, "stok": 45,
        "warna": ["Putih"], "kategori": "Aksesoris",
    },
}

DEFAULT_ORDERS = {
    "ORD-10001": {
        "produk": "iPhone 15 Pro", "status": "Dalam Pengiriman",
        "kurir": "JNE REG", "resi": "JNE8827361524",
        "estimasi": "12 April 2026", "alamat": "Jakarta Selatan",
    },
    "ORD-10002": {
        "produk": "Samsung Galaxy S24", "status": "Sedang Dikemas",
        "kurir": "-", "resi": "-", "estimasi": "dikirim besok", "alamat": "Bandung",
    },
    "ORD-10003": {
        "produk": "AirPods Pro 2", "status": "Sudah Diterima",
        "kurir": "SiCepat", "resi": "SCP992817364",
        "estimasi": "-", "alamat": "Surabaya",
    },
    "ORD-10004": {
        "produk": "MacBook Air M3", "status": "Menunggu Pembayaran",
        "kurir": "-", "resi": "-",
        "estimasi": "proses setelah bayar", "alamat": "Yogyakarta",
    },
}

if "product_db" not in st.session_state:
    st.session_state.product_db = DEFAULT_PRODUCTS.copy()
if "order_db" not in st.session_state:
    st.session_state.order_db = DEFAULT_ORDERS.copy()


# ============================================================
# CSV UTILITIES
# ============================================================

def decode_csv_bytes(raw_bytes):
    """Decode file CSV dengan berbagai encoding (UTF-8, UTF-16, Excel BOM)."""
    if raw_bytes[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw_bytes.decode("utf-16")
    
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            text = raw_bytes.decode(enc)
            return text.replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("latin-1").replace("\x00", "")


def detect_csv_delimiter(text):
    """Auto-detect delimiter CSV (comma, semicolon, tab)."""
    lines = text.strip().split("\n")
    
    # Excel kadang tambahkan 'sep=;' di baris pertama
    start = 0
    if lines[0].strip().lower().startswith("sep="):
        start = 1
    
    if len(lines) <= start:
        return ",", start
    
    # Pakai csv.Sniffer dulu -- paling reliable
    test_text = "\n".join(lines[start:start+3])
    try:
        dialect = csv.Sniffer().sniff(test_text, delimiters=',;\t|')
        return dialect.delimiter, start
    except csv.Error:
        pass
    
    # Fallback: hitung delimiter terbanyak di header
    header = lines[start]
    for delim in [",", ";", "\t", "|"]:
        if header.count(delim) >= 2:
            return delim, start
    
    return ",", start


def parse_product_csv(file_content):
    """
    Parse CSV produk. Kolom yang didukung: nama, harga, stok, warna, kategori.
    Handle berbagai format dari Excel, Google Sheets, atau text editor.
    """
    if file_content.startswith("\ufeff"):
        file_content = file_content[1:]
    
    file_content = file_content.replace("\x00", "")
    file_content = file_content.replace("\r\n", "\n").replace("\r", "\n")
    
    lines = [l for l in file_content.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return {}
    
    delimiter, skip = detect_csv_delimiter(file_content)
    lines = lines[skip:]
    if len(lines) < 2:
        return {}
    
    # Parse header dan mapping kolom
    header_reader = csv.reader([lines[0]], delimiter=delimiter)
    headers = [h.strip().lower() for h in next(header_reader)]
    
    col_map = {}
    for i, h in enumerate(headers):
        if "nama" in h or "name" in h or "produk" in h:
            col_map["nama"] = i
        elif "harga" in h or "price" in h:
            col_map["harga"] = i
        elif "stok" in h or "stock" in h or "qty" in h:
            col_map["stok"] = i
        elif "warna" in h or "color" in h or "varian" in h:
            col_map["warna"] = i
        elif "kategori" in h or "category" in h or "tipe" in h:
            col_map["kategori"] = i
    
    if "nama" not in col_map:
        col_map = {"nama": 0, "harga": 1, "stok": 2, "warna": 3, "kategori": 4}
    
    products = {}
    reader = csv.reader(lines[1:], delimiter=delimiter)
    
    for row in reader:
        if not row:
            continue
        
        # Fix: Excel kadang wrap seluruh baris dalam quotes
        expected_cols = max(col_map.values()) + 1 if col_map else 5
        if len(row) == 1 and expected_cols > 1:
            reparsed = list(csv.reader([row[0]], delimiter=delimiter))
            if reparsed and len(reparsed[0]) > 1:
                row = reparsed[0]
        
        def get_col(name, default=""):
            idx = col_map.get(name)
            if idx is not None and idx < len(row):
                return row[idx].strip()
            return default
        
        nama = get_col("nama")
        if not nama:
            continue
        
        warna_raw = get_col("warna")
        warna = [w.strip() for w in warna_raw.split(",") if w.strip()] if warna_raw else []
        
        try:
            harga = int(float(get_col("harga", "0") or "0"))
        except (ValueError, TypeError):
            harga = 0
        try:
            stok = int(float(get_col("stok", "0") or "0"))
        except (ValueError, TypeError):
            stok = 0
        
        products[nama.lower()] = {
            "nama": nama,
            "harga": harga,
            "stok": stok,
            "warna": warna,
            "kategori": get_col("kategori", "Lainnya") or "Lainnya",
        }
    return products


def parse_order_csv(file_content):
    """Parse CSV pesanan. Kolom: nomor_order, produk, status, kurir, resi, estimasi, alamat."""
    if file_content.startswith("\ufeff"):
        file_content = file_content[1:]
    file_content = file_content.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    
    delimiter, skip = detect_csv_delimiter(file_content)
    lines = [l for l in file_content.strip().split("\n") if l.strip()]
    lines = lines[skip:]
    if len(lines) < 2:
        return {}
    
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    orders = {}
    for row in reader:
        row = {(k or "").strip().lower(): v for k, v in row.items()}
        key = (row.get("nomor_order") or "").strip().upper()
        if not key:
            continue
        orders[key] = {
            "produk": (row.get("produk") or "").strip(),
            "status": (row.get("status") or "").strip(),
            "kurir": (row.get("kurir") or "-").strip(),
            "resi": (row.get("resi") or "-").strip(),
            "estimasi": (row.get("estimasi") or "-").strip(),
            "alamat": (row.get("alamat") or "-").strip(),
        }
    return orders


def generate_product_template():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["nama", "harga", "stok", "warna", "kategori"])
    writer.writerow(["Contoh Produk A", "15000000", "10", "Hitam, Putih, Biru", "Elektronik"])
    writer.writerow(["Contoh Produk B", "5000000", "25", "Merah, Hijau", "Aksesoris"])
    return output.getvalue()


def generate_order_template():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["nomor_order", "produk", "status", "kurir", "resi", "estimasi", "alamat"])
    writer.writerow(["ORD-00001", "Contoh Produk A", "Dalam Pengiriman", "JNE", "JNE123456", "3 hari lagi", "Jakarta"])
    writer.writerow(["ORD-00002", "Contoh Produk B", "Sedang Dikemas", "-", "-", "dikirim besok", "Surabaya"])
    return output.getvalue()


# ============================================================
# AGENT TOOLS
# ============================================================

def cari_produk(query: str) -> str:
    """Cari produk berdasarkan nama atau kategori."""
    results = []
    q = query.lower()
    for key, prod in st.session_state.product_db.items():
        if q in key or q in prod["nama"].lower() or q in prod.get("kategori", "").lower():
            results.append({
                "nama": prod["nama"],
                "harga": f"Rp {prod.get('harga', 0):,}".replace(",", "."),
                "stok": prod.get("stok", 0),
                "status": "Tersedia" if prod.get("stok", 0) > 0 else "Habis",
                "warna": prod.get("warna", []),
                "kategori": prod.get("kategori", "-"),
            })
    
    if results:
        return json.dumps({"ditemukan": len(results), "produk": results}, ensure_ascii=False)
    return json.dumps({"ditemukan": 0, "pesan": f"Produk '{query}' tidak ditemukan di katalog."}, ensure_ascii=False)


def hitung_diskon(harga: float, persen_diskon: float) -> str:
    """Hitung harga setelah diskon."""
    potongan = harga * (persen_diskon / 100)
    final = harga - potongan
    return json.dumps({
        "harga_asli": f"Rp {harga:,.0f}".replace(",", "."),
        "diskon": f"{persen_diskon}%",
        "potongan": f"Rp {potongan:,.0f}".replace(",", "."),
        "harga_final": f"Rp {final:,.0f}".replace(",", "."),
    }, ensure_ascii=False)


def cek_pesanan(nomor_order: str) -> str:
    """Cek status pesanan berdasarkan nomor order."""
    order = st.session_state.order_db.get(nomor_order.upper())
    if order:
        return json.dumps({"nomor_order": nomor_order, **order}, ensure_ascii=False)
    return json.dumps({
        "nomor_order": nomor_order,
        "status": "Tidak Ditemukan",
        "pesan": "Nomor order tidak valid. Format: ORD-XXXXX"
    }, ensure_ascii=False)


def lihat_katalog(kategori: str = "") -> str:
    """Lihat semua produk atau filter berdasarkan kategori."""
    results = []
    for key, prod in st.session_state.product_db.items():
        if not kategori or kategori.lower() in prod.get("kategori", "").lower():
            results.append({
                "nama": prod["nama"],
                "harga": f"Rp {prod.get('harga', 0):,}".replace(",", "."),
                "stok": prod.get("stok", 0),
                "status": "Tersedia" if prod.get("stok", 0) > 0 else "Habis",
                "kategori": prod.get("kategori", "-"),
            })
    return json.dumps({"total_produk": len(results), "katalog": results}, ensure_ascii=False)


# -- RAG: Document search tool --

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def process_uploaded_docs(files):
    """Proses dokumen PDF/TXT menjadi vector store untuk pencarian."""
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path) if file.name.endswith(".pdf") else TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    chunks = splitter.split_documents(all_docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    
    if not chunks:
        return None, 0
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=get_embeddings())
    return vectorstore, len(chunks)


def cari_di_dokumen(query: str) -> str:
    """Cari informasi di dokumen yang sudah diupload."""
    vs = st.session_state.get("doc_vectorstore")
    if not vs:
        return json.dumps({"error": "Tidak ada dokumen yang diupload."}, ensure_ascii=False)
    
    results = vs.similarity_search(query, k=5)
    if not results:
        return json.dumps({"ditemukan": 0, "pesan": f"Tidak ada info tentang '{query}' di dokumen."}, ensure_ascii=False)
    
    docs = [{
        "sumber": doc.metadata.get("source", "?"),
        "halaman": doc.metadata.get("page", "?"),
        "isi": doc.page_content[:500],
    } for doc in results]
    return json.dumps({"ditemukan": len(docs), "hasil": docs}, ensure_ascii=False)


# Tool registry
TOOL_MAP = {
    "cari_produk": cari_produk,
    "hitung_diskon": hitung_diskon,
    "cek_pesanan": cek_pesanan,
    "lihat_katalog": lihat_katalog,
    "cari_di_dokumen": cari_di_dokumen,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "cari_produk",
            "description": "Cari produk berdasarkan nama atau kata kunci.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Nama produk atau kata kunci"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hitung_diskon",
            "description": "Hitung harga setelah diskon.",
            "parameters": {
                "type": "object",
                "properties": {
                    "harga": {"type": "number", "description": "Harga asli dalam Rupiah"},
                    "persen_diskon": {"type": "number", "description": "Persentase diskon (contoh: 20 untuk 20%)"}
                },
                "required": ["harga", "persen_diskon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cek_pesanan",
            "description": "Cek status pengiriman pesanan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nomor_order": {"type": "string", "description": "Nomor order (format: ORD-XXXXX)"}
                },
                "required": ["nomor_order"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lihat_katalog",
            "description": "Lihat semua produk atau filter berdasarkan kategori.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kategori": {"type": "string", "description": "Filter kategori (opsional)", "default": ""}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cari_di_dokumen",
            "description": "Cari informasi di dokumen yang sudah diupload (PDF/TXT).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Kata kunci atau pertanyaan untuk dicari di dokumen"}
                },
                "required": ["query"]
            }
        }
    },
]


# ============================================================
# LLM PROVIDERS
# ============================================================

PROVIDERS = {
    "Groq (Gratis)": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "placeholder": "gsk_...",
        "signup": "https://console.groq.com",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "placeholder": "sk-...",
        "signup": "https://platform.openai.com",
    },
    "Google Gemini (Gratis)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"],
        "placeholder": "AIza...",
        "signup": "https://ai.google.dev",
    },
    "OpenRouter (Gratis)": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-8b:free",
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
        help=f"Daftar gratis di {pcfg['signup']}"
    )
    model_name = st.selectbox("Model", pcfg["models"])
    
    st.divider()
    st.subheader("Fitur Agent")
    show_reasoning = st.checkbox("Tampilkan proses berpikir AI", value=True,
        help="Lihat tool apa yang dipanggil agent dan hasilnya")
    
    st.divider()
    st.subheader("Contoh Pertanyaan")
    examples = [
        "iPhone 15 masih ada stok?",
        "Cek pesanan ORD-10001",
        "Hitung diskon 25% dari 15 juta",
        "Tampilkan semua produk laptop",
        "Cari di dokumen tentang kebijakan retur",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state["prefill"] = ex
    
    # Upload dokumen untuk RAG
    st.divider()
    st.subheader("Upload Dokumen")
    doc_files = st.file_uploader(
        "PDF atau TXT", type=["pdf", "txt"],
        accept_multiple_files=True, key="doc_upload"
    )
    if doc_files:
        doc_names = sorted([f.name for f in doc_files])
        if st.session_state.get("doc_file_names") != doc_names:
            with st.spinner("Memproses dokumen..."):
                vs, n_chunks = process_uploaded_docs(doc_files)
                if vs:
                    st.session_state.doc_vectorstore = vs
                    st.session_state.doc_file_names = doc_names
                    st.success(f"{len(doc_names)} dokumen ({n_chunks} chunks)")
                else:
                    st.error("Tidak ada teks yang bisa di-extract.")
    elif "doc_vectorstore" in st.session_state:
        st.caption(f"Dokumen aktif: {len(st.session_state.get('doc_file_names', []))} file")
    
    # Upload data custom (produk & pesanan)
    st.divider()
    st.subheader("Data Custom (Produk/Pesanan)")
    
    data_tab = st.radio("Sumber Data", ["Default (Demo)", "Upload CSV"], horizontal=True)
    
    if data_tab == "Upload CSV":
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("Template Produk", generate_product_template(),
                "template_produk.csv", "text/csv", use_container_width=True)
        with col_b:
            st.download_button("Template Pesanan", generate_order_template(),
                "template_pesanan.csv", "text/csv", use_container_width=True)
        
        product_file = st.file_uploader("Data Produk (CSV)", type=["csv"], key="prod_csv")
        if product_file:
            try:
                content = decode_csv_bytes(product_file.getvalue())
                parsed = parse_product_csv(content)
                if parsed:
                    st.session_state.product_db = parsed
                    st.success(f"Loaded {len(parsed)} produk")
                    for v in list(parsed.values())[:3]:
                        harga = f"Rp {v['harga']:,}".replace(",", ".")
                        st.caption(f"  {v['nama']} | {harga} | {v['stok']} stok")
                else:
                    st.warning("CSV terbaca tapi tidak ada data.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        
        order_file = st.file_uploader("Data Pesanan (CSV)", type=["csv"], key="order_csv")
        if order_file:
            try:
                content = decode_csv_bytes(order_file.getvalue())
                parsed = parse_order_csv(content)
                if parsed:
                    st.session_state.order_db = parsed
                    st.success(f"Loaded {len(parsed)} pesanan")
                else:
                    st.warning("CSV terbaca tapi tidak ada data.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
    else:
        if st.button("Reset ke Data Demo", use_container_width=True):
            st.session_state.product_db = DEFAULT_PRODUCTS.copy()
            st.session_state.order_db = DEFAULT_ORDERS.copy()
            st.success("Data direset ke default")
    
    st.caption(f"Produk: {len(st.session_state.product_db)} | Pesanan: {len(st.session_state.order_db)}")
    
    st.divider()
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_logs = []
        st.rerun()


# ============================================================
# AGENT CORE
# ============================================================

SYSTEM_PROMPT = """Kamu adalah TeknoBot, customer service AI di toko online TeknoShop.

KEMAMPUAN:
- Cari produk menggunakan tool cari_produk
- Hitung diskon menggunakan tool hitung_diskon
- Cek status pesanan menggunakan tool cek_pesanan
- Tampilkan katalog menggunakan tool lihat_katalog
- Cari di dokumen menggunakan tool cari_di_dokumen

ATURAN:
- WAJIB gunakan tools untuk pertanyaan tentang produk, harga, stok, atau pesanan
- DILARANG mengarang data produk, harga, atau stok
- HANYA sebutkan produk yang muncul dari hasil tools
- Jika produk tidak ditemukan, bilang jujur

GAYA:
- Panggil customer dengan "Kak"
- Ramah, profesional, informatif
- Jawab dalam Bahasa Indonesia"""


def run_agent_step(messages, api_key, base_url, model):
    """
    Jalankan agent loop (ReAct pattern).
    Agent bisa memanggil tools berulang kali sampai punya cukup info untuk menjawab.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    tool_logs = []
    
    for iteration in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as e:
            error_msg = str(e)
            if "tool" in error_msg.lower() or "function" in error_msg.lower():
                # Fallback: inject data produk langsung ke prompt
                tool_logs.append({
                    "step": iteration + 1,
                    "action": "fallback",
                    "detail": "Tool calling tidak didukung, menggunakan data lokal",
                })
                
                db = st.session_state.get("product_db", {})
                product_list = [
                    f"- {v['nama']}: Rp {v.get('harga', 0):,}, stok {v.get('stok', 0)}".replace(",", ".")
                    for v in db.values()
                ]
                data_ctx = "\n".join(product_list) if product_list else "Tidak ada data."
                
                fallback_messages = messages.copy()
                fallback_messages[0] = {"role": "system", "content": f"""Kamu adalah TeknoBot, customer service AI.
DATA PRODUK (HANYA jawab dari data ini, JANGAN mengarang):
{data_ctx}

Jawab dalam Bahasa Indonesia. Panggil customer dengan "Kak"."""}
                
                response = client.chat.completions.create(
                    model=model, messages=fallback_messages, temperature=0.1,
                )
                return response.choices[0].message.content, tool_logs
            else:
                raise e
        
        msg = response.choices[0].message
        
        # Kalau ada tool calls, jalankan tools
        if msg.tool_calls:
            messages.append(msg)
            
            for call in msg.tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)
                
                if func_name in TOOL_MAP:
                    result = TOOL_MAP[func_name](**func_args)
                else:
                    result = json.dumps({"error": f"Tool '{func_name}' tidak tersedia"})
                
                tool_logs.append({
                    "step": iteration + 1,
                    "tool": func_name,
                    "args": func_args,
                    "result": result,
                })
                
                messages.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "content": result,
                })
        else:
            # Agent sudah selesai berpikir, return jawaban
            return msg.content, tool_logs
    
    return "Maaf, saya tidak bisa memproses permintaan saat ini. Silakan coba lagi.", tool_logs


# ============================================================
# MAIN APP
# ============================================================

if not api_key:
    st.info("Masukkan API Key di sidebar untuk mulai chat dengan TeknoBot.")
    
    # Tampilkan katalog produk sebagai landing page
    st.markdown("### Katalog Produk")
    db = st.session_state.product_db
    cols = st.columns(4)
    for i, (key, prod) in enumerate(db.items()):
        with cols[i % 4]:
            status = "Tersedia" if prod.get("stok", 0) > 0 else "Habis"
            badge_class = "badge-green" if prod.get("stok", 0) > 0 else "badge-red"
            harga = f"Rp {prod.get('harga', 0):,}".replace(",", ".")
            st.markdown(f"""
            <div class="product-card">
                <h4>{prod['nama']}</h4>
                <p class="price">{harga}</p>
                <p>Stok: {prod.get('stok', 0)} unit</p>
                <span class="badge {badge_class}">{status}</span>
                <span class="badge badge-blue">{prod.get('kategori', '-')}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.stop()

# Stats bar
db_prod = st.session_state.product_db
db_ord = st.session_state.order_db
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="stat-box"><h3>{len(db_prod)}</h3><p>Produk</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-box"><h3>{len(db_ord)}</h3><p>Pesanan</p></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-box"><h3>{len(TOOLS_SCHEMA)}</h3><p>Tools</p></div>', unsafe_allow_html=True)
provider_short = provider.split(" (")[0]
c4.markdown(f'<div class="stat-box"><h3>{provider_short}</h3><p>AI Provider</p></div>', unsafe_allow_html=True)

st.divider()


# ============================================================
# CHAT INTERFACE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_logs" not in st.session_state:
    st.session_state.tool_logs = []


def render_tool_logs(logs):
    """Render tool call logs sebagai HTML."""
    for log in logs:
        if "tool" in log:
            st.markdown(f"""
            <div class="tool-log">
                Step {log['step']}: Memanggil <span class="tool-name">{log['tool']}</span><br>
                Input: <span class="tool-args">{json.dumps(log['args'], ensure_ascii=False)}</span><br>
                Output: <span class="tool-result">{log['result'][:200]}...</span>
            </div>
            """, unsafe_allow_html=True)
        elif log.get("action") == "fallback":
            st.markdown(f"""
            <div class="reasoning-step">{log['detail']}</div>
            """, unsafe_allow_html=True)


# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    
    if msg["role"] == "assistant" and show_reasoning:
        logs = [l for l in st.session_state.tool_logs if l.get("msg_index") == i]
        if logs:
            with st.expander(f"Proses Berpikir ({len(logs)} langkah)"):
                render_tool_logs(logs)

# Handle input
prefill = st.session_state.pop("prefill", None)

if user_input := (prefill or st.chat_input("Tanya TeknoBot tentang produk, pesanan, atau diskon...")):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("TeknoBot sedang berpikir..."):
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in st.session_state.messages[-10:]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            try:
                answer, logs = run_agent_step(
                    api_messages, api_key, pcfg["base_url"], model_name
                )
                
                st.markdown(answer)
                
                # Simpan logs
                msg_index = len(st.session_state.messages)
                for log in logs:
                    log["msg_index"] = msg_index
                st.session_state.tool_logs.extend(logs)
                
                if show_reasoning and logs:
                    with st.expander(f"Proses Berpikir ({len(logs)} langkah)"):
                        render_tool_logs(logs)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
