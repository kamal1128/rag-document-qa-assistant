from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("Document loaders/GRU.pdf")

docs = data.load()

print(docs[0].page_content)