import os
import hashlib
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="DocuMind | AI Knowledge Workspace",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- DESIGN SYSTEM ---------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#080d18;--panel:#101a2b;--line:rgba(159,184,220,.14);--muted:#91a2bc;--text:#edf4ff;--cyan:#69d9e8;--violet:#a78bfa}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(55,110,190,.19),transparent 34%),radial-gradient(ellipse at 90% 12%,rgba(132,83,210,.15),transparent 30%),linear-gradient(145deg,#080d18 0%,#0b1322 52%,#080d18 100%);color:var(--text)}
[data-testid="stHeader"]{background:rgba(8,13,24,.78)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1728 0%,#0a1220 100%);border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div{padding-top:1.2rem}
.block-container{max-width:1440px;padding-top:2rem;padding-bottom:7rem}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.035em}
.brand-row{display:flex;align-items:center;gap:12px;margin:0 0 1.3rem}
.brand-mark{width:43px;height:43px;border-radius:14px;display:flex;align-items:center;justify-content:center;color:#07131d;font-size:23px;font-weight:900;background:linear-gradient(135deg,#8ce8ee,#9a8cff);box-shadow:0 8px 30px rgba(105,217,232,.22)}
.brand-name{font:800 20px 'Manrope',sans-serif;color:#f3f7ff}.brand-sub{color:#8193ae;font-size:10px;letter-spacing:.17em;text-transform:uppercase}
.hero{position:relative;overflow:hidden;padding:31px 35px;border:1px solid rgba(149,177,220,.19);border-radius:24px;background:linear-gradient(115deg,rgba(22,42,69,.96),rgba(21,28,53,.88));box-shadow:0 20px 70px rgba(0,0,0,.22);animation:rise .65s ease both}
.hero:before{content:"";position:absolute;width:320px;height:320px;right:-85px;top:-180px;border-radius:50%;background:rgba(104,219,232,.14);filter:blur(5px);animation:float 8s ease-in-out infinite}
.hero:after{content:"";position:absolute;width:190px;height:190px;right:155px;bottom:-150px;border-radius:50%;background:rgba(167,139,250,.19);filter:blur(8px);animation:float 10s ease-in-out infinite reverse}
.eyebrow{position:relative;z-index:1;color:#8edee9;font-size:10px;font-weight:700;letter-spacing:.21em;text-transform:uppercase;margin-bottom:13px}
.hero-title{position:relative;z-index:1;color:#f5f8ff;font:800 clamp(29px,4vw,45px) 'Manrope',sans-serif;line-height:1.12;letter-spacing:-.055em;margin:0;max-width:760px}
.hero-title span{background:linear-gradient(90deg,#8de7eb,#b7a4ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero-copy{position:relative;z-index:1;color:#aabbd3;font-size:14px;margin-top:13px;max-width:650px;line-height:1.75}
.hero-pill{position:relative;z-index:1;display:inline-flex;align-items:center;gap:9px;margin-top:20px;padding:8px 12px;border-radius:99px;color:#c9d8eb;border:1px solid rgba(145,190,220,.22);background:rgba(7,15,28,.35);font-size:10px;letter-spacing:.07em}
.pulse-dot{width:7px;height:7px;border-radius:50%;background:#72e0b2;box-shadow:0 0 0 5px rgba(114,224,178,.1);animation:pulse 2s infinite}
.section-label{color:#8498b6;font-size:10px;font-weight:700;letter-spacing:.19em;text-transform:uppercase;margin:20px 0 11px 2px}
.metric-card{border:1px solid var(--line);border-radius:17px;padding:17px 18px;background:linear-gradient(145deg,rgba(20,35,57,.94),rgba(14,24,41,.94));min-height:105px;transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}
.metric-card:hover{transform:translateY(-4px);border-color:rgba(105,217,232,.45);box-shadow:0 12px 30px rgba(0,0,0,.2)}
.metric-label{color:#8fa2bf;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.metric-value{color:#f4f7ff;font:800 27px 'Manrope',sans-serif;margin-top:8px}.metric-foot{color:#8193ad;font-size:11px;margin-top:3px}
.panel{border:1px solid var(--line);border-radius:19px;background:linear-gradient(145deg,rgba(17,29,48,.94),rgba(13,23,39,.96));padding:19px 20px}
.panel-title{color:#eef4ff;font:700 15px 'Manrope',sans-serif;margin-bottom:5px}.panel-copy{color:#8fa2bd;font-size:12px;line-height:1.7}
[data-testid="stChatMessage"]{border:1px solid rgba(145,170,210,.14);border-radius:17px;padding:13px 15px;background:rgba(16,27,45,.84);animation:rise .3s ease both}
[data-testid="stChatInput"]{border:1px solid rgba(139,170,215,.25)!important;border-radius:17px!important;background:rgba(15,26,44,.97)!important;box-shadow:0 12px 38px rgba(0,0,0,.22)}
[data-testid="stChatInput"] textarea{color:#f3f7ff!important}
.stButton>button{border-radius:12px;border:1px solid rgba(133,177,220,.2);background:linear-gradient(110deg,#6bd9e6,#a99aff);color:#07131d;font-weight:800;min-height:43px;transition:all .2s ease}
.stButton>button:hover{border-color:#a8f0f4;transform:translateY(-2px);box-shadow:0 8px 24px rgba(105,217,232,.18)}
.stButton>button:disabled{opacity:.4}
div[data-testid="stFileUploader"]{border:1px dashed rgba(133,177,220,.38);border-radius:15px;padding:8px;background:rgba(17,30,50,.5)}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#68dce7,#a78bfa)}
hr{border-color:rgba(145,170,210,.12)!important}
.small-note{color:#8193ad;font-size:11px;line-height:1.65}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(22px)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 4px rgba(114,224,178,.1)}50%{box-shadow:0 0 0 8px rgba(114,224,178,.02)}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>
""", unsafe_allow_html=True)

# --------------------------- CONFIG ---------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GROQ_MODEL = "openai/gpt-oss-20b"


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_model(),
    )


@st.cache_resource
def get_llm(api_key):
    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key)


def extract_and_split_pdf(uploaded_file):
    raw = uploaded_file.getvalue()
    file_hash = hashlib.sha256(raw).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(raw)
        temp_path = temp.name
    try:
        pages = PyPDFLoader(temp_path).load()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    for page in pages:
        page.metadata["source"] = uploaded_file.name
        page.metadata["source_hash"] = file_hash
    if not any(page.page_content.strip() for page in pages):
        raise ValueError("No selectable text found. This PDF may be scanned and require OCR.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    return chunks, file_hash, len(pages)


def index_pdf(uploaded_file, db):
    chunks, file_hash, page_count = extract_and_split_pdf(uploaded_file)
    existing = db.get(where={"source_hash": file_hash}, include=["metadatas"])
    if existing.get("ids"):
        return {"status": "duplicate", "name": uploaded_file.name}
    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    db.add_documents(documents=chunks, ids=ids)
    return {"status": "added", "name": uploaded_file.name, "pages": page_count, "chunks": len(chunks)}


def get_retriever(db):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def make_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page")
        page_text = f", page {int(page) + 1}" if page is not None else ""
        parts.append(f"[Source {i}: {source}{page_text}]\n{doc.page_content}")
    return "\n\n".join(parts)


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocuMind, a precise document question-answering assistant.
Use only the supplied context. If the answer is not supported by the context, say:
'I couldn't find that information in the uploaded documents.' Do not invent facts.
Keep answers clear and structured. Cite the source filename and page when available."""),
    ("human", "DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"),
])


def answer_question(question, db, llm):
    docs = get_retriever(db).invoke(question)
    if not docs:
        return "I couldn't find relevant information in the indexed documents.", []
    prompt_value = PROMPT.invoke({"context": make_context(docs), "question": question})
    return llm.invoke(prompt_value).content, docs


# --------------------------- SESSION STATE ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_report" not in st.session_state:
    st.session_state.upload_report = None

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">✦</div><div>'
        '<div class="brand-name">DocuMind</div><div class="brand-sub">AI knowledge workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Knowledge library</div>', unsafe_allow_html=True)
    files = st.file_uploader("Add PDF documents", type=["pdf"], accept_multiple_files=True)
    if st.button("＋  Index documents", type="primary", use_container_width=True, disabled=not files):
        try:
            db = get_vectorstore()
            progress = st.progress(0, text="Preparing ingestion...")
            status = st.empty()
            added, duplicates, errors = [], [], []
            for i, file in enumerate(files):
                status.caption(f"Processing {file.name} · {i + 1} of {len(files)}")
                try:
                    result = index_pdf(file, db)
                    if result["status"] == "added":
                        added.append(result)
                    else:
                        duplicates.append(file.name)
                except Exception as exc:
                    errors.append((file.name, str(exc)))
                progress.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)} files")
            progress.empty()
            status.empty()
            st.session_state.upload_report = {"added": added, "duplicates": duplicates, "errors": errors}
            if added:
                st.success(f"Indexed {len(added)} new PDF(s).")
                for item in added:
                    st.caption(f"✓ {item['name']} · {item['pages']} pages · {item['chunks']} chunks")
                st.session_state.messages.append({"role": "assistant", "content": "Your documents are indexed. Ask me anything about their contents."})
            if duplicates:
                st.info("Already indexed: " + ", ".join(duplicates))
            for name, error in errors:
                st.error(f"{name}: {error}")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    st.markdown("---")
    st.markdown('<div class="section-label">System configuration</div>', unsafe_allow_html=True)
    try:
        db = get_vectorstore()
        chunk_count = db._collection.count()
    except Exception:
        chunk_count = 0
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Indexed chunks</div><div class="metric-value">{chunk_count:,}</div><div class="metric-foot">Stored in local ChromaDB</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><div class="panel-title">Retrieval settings</div><div class="panel-copy">'
        'Embedding · all-mpnet-base-v2<br>Search · MMR<br>Top-K · 4 chunks<br>Chunk size · 1,000 characters</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('<p class="small-note">Clearing chat does not delete indexed documents.</p>', unsafe_allow_html=True)

# --------------------------- MAIN HEADER ---------------------------
st.markdown(
    '<div class="hero"><div class="eyebrow">PERSONAL DOCUMENT INTELLIGENCE</div>'
    '<h1 class="hero-title">Your knowledge, <span>within reach.</span></h1>'
    '<div class="hero-copy">A calm, focused workspace to explore PDFs, find precise answers, '
    'and trace every response back to its source.</div>'
    '<div class="hero-pill"><span class="pulse-dot"></span> LOCAL DOCUMENT INDEX &nbsp;·&nbsp; GROQ-POWERED ANSWERS</div></div>',
    unsafe_allow_html=True,
)

try:
    main_db = get_vectorstore()
    total_chunks = main_db._collection.count()
except Exception:
    main_db, total_chunks = None, 0
api_key = get_api_key()

st.markdown('<div class="section-label">Workspace overview</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Knowledge indexed</div><div class="metric-value">{total_chunks:,}</div><div class="metric-foot">Chunks available for retrieval</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Retrieval strategy</div><div class="metric-value">MMR</div><div class="metric-foot">Relevance balanced with diversity</div></div>', unsafe_allow_html=True)
with c3:
    color = "#79e0b4" if api_key else "#f3bd73"
    label = "Connected" if api_key else "Key required"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Generation engine</div><div class="metric-value" style="font-size:23px;color:{color}">{label}</div><div class="metric-foot">Groq · {GROQ_MODEL}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Document conversation</div>', unsafe_allow_html=True)
if not api_key:
    st.warning("GROQ_API_KEY not found. Add it to your .env file or Streamlit Secrets to enable chat.")

if not st.session_state.messages:
    st.markdown('<div class="panel"><div class="panel-title">Start exploring your documents</div><div class="panel-copy">Ask for a definition, summary, comparison, or explanation. DocuMind retrieves relevant passages and grounds its answers in your PDFs.</div></div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your indexed documents…", disabled=not bool(api_key))
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            db = get_vectorstore()
            if db._collection.count() == 0:
                answer, docs = "Your knowledge base is empty. Upload and index a PDF first.", []
            else:
                llm = get_llm(api_key)
                with st.spinner("Searching your knowledge base and composing an answer…"):
                    answer, docs = answer_question(question, db, llm)
            st.markdown(answer)
            if docs:
                with st.expander("Sources used for this answer"):
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "Unknown document")
                        page = doc.metadata.get("page")
                        page_label = f" · Page {int(page) + 1}" if page is not None else ""
                        st.markdown(f"**{i}. {source}{page_label}**")
                        st.write(doc.page_content)
                        if i < len(docs):
                            st.divider()
        except Exception as exc:
            answer = "I couldn't complete that request. Check the technical details below."
            st.error(answer)
            st.exception(exc)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown('<div style="text-align:center;color:#63748f;font-size:10px;letter-spacing:.14em;padding-top:28px">DOCUMIND · DOCUMENT INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
