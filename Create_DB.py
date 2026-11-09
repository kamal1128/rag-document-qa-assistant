#Pypdf
#resusrivecharectertextsplitter
#embeddings
#chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#Document Loaders
data = PyPDFLoader("Document loaders/deeplearning.pdf")
docs = data.load()

#Text Splitters/Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunk = splitter.split_documents(docs)

#Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#chroma_db
vectorstore = Chroma.from_documents(
    documents = chunk,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)