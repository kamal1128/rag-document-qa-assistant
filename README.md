# 📚 RAG Document Q&A Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their content using natural language.

Built using Python, LangChain, Hugging Face Embeddings, ChromaDB, Groq LLM, and Streamlit.

## 🚀 Live Demo

**[Try the RAG Document Q&A Assistant](https://rag-document-app-assistant-7gk4ebsob5cmwwzbaryzju.streamlit.app/)**

## ✨ Features

- **PDF Upload:** Upload one or multiple PDF documents.
- **Document Processing:** Extract text from PDFs using PyPDFLoader.
- **Text Chunking:** Split documents using RecursiveCharacterTextSplitter.
- **Semantic Search:** Retrieve relevant document chunks using vector embeddings.
- **Vector Database:** Store and search embeddings using ChromaDB.
- **MMR Retrieval:** Improve retrieval diversity and reduce redundant context.
- **AI-Powered Answers:** Generate context-aware responses using Groq LLM.
- **Source References:** Display retrieved document content and page information.
- **Interactive UI:** Chat with your uploaded documents through Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | RAG pipeline orchestration |
| PyPDF | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| Hugging Face | Text embeddings |
| all-mpnet-base-v2 | Embedding model |
| ChromaDB | Vector database |
| Groq | LLM inference |
| Git & GitHub | Version control |

## 🏗️ RAG Architecture

```text
             PDF Upload
                  |
                  v
          PyPDFLoader
                  |
                  v
        Text Extraction
                  |
                  v
   Recursive Character Text Splitter
                  |
                  v
       Document Chunking
                  |
                  v
      Hugging Face Embeddings
                  |
                  v
             ChromaDB
                  |
           Vector Storage
                  |
          USER QUESTION
                  |
                  v
          Query Embedding
                  |
                  v
          MMR Retrieval
                  |
                  v
       Relevant Document Chunks
                  |
                  v
          Prompt Template
                  |
                  v
             Groq LLM
                  |
                  v
           Final Answer
```

## 🧠 How It Works

### 1. Document Loading
PDF documents are uploaded and processed using PyPDFLoader to extract their text.

### 2. Text Splitting
RecursiveCharacterTextSplitter divides extracted text into smaller chunks.

Current configuration:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### 3. Embeddings
The Hugging Face model `sentence-transformers/all-mpnet-base-v2` converts document chunks into numerical vector representations.

### 4. Vector Storage
ChromaDB stores document embeddings, text chunks, and metadata for efficient similarity search.

### 5. Retrieval
Maximum Marginal Relevance (MMR) retrieves relevant and diverse document chunks based on the user's query.

Retrieval configuration:

```python
search_kwargs = {
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

### 6. LLM Generation
The retrieved context and user question are passed to Groq's `openai/gpt-oss-20b` model through LangChain's ChatGroq integration.

The LLM generates an answer grounded in the retrieved document context.

## 📂 Project Structure

```text
rag-document-qa-assistant/
│
├── app.py
├── main.py
├── Create_DB.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
│
├── Document loaders/
│   └── deeplearning.pdf
│
└── chroma_db/
```

*Note: `.env`, `.venv`, and generated ChromaDB files are excluded from GitHub.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kamal1128/rag-document-qa-assistant.git
```

### 2. Navigate to the project directory

```bash
cd rag-document-qa-assistant
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows Git Bash:

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

**Never upload your `.env` file or expose your API key on GitHub.**

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL provided by Streamlit.

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| GROQ_API_KEY | API key used for Groq LLM access |

For Streamlit Cloud deployment, configure the API key under the application's Secrets settings instead of uploading `.env`.

## 🎯 Future Improvements

- Conversation memory for follow-up questions.
- Support for DOCX, TXT, and other document formats.
- Hybrid search combining BM25 and vector retrieval.
- Cross-encoder reranking.
- RAG evaluation using RAGAS.
- Improved document ingestion and caching.
- Persistent cloud vector storage.

## 👨‍💻 Author

**Sai Kamal Kandukuri**

- GitHub: [kamal1128](https://github.com/kamal1128)
- LinkedIn: [Sai Kamal Kandukuri](https://www.linkedin.com/in/sai-kamal-kandukuri-404288305)
- Portfolio: [My Portfolio](https://neon-kamal-site.lovable.app/)

---

⭐ If you find this project useful, consider giving it a star on GitHub.
