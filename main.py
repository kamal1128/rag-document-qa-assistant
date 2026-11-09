import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstor = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstor.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" : 0.5
    }
 
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpfl AI assistant. 
         Use ONLY the provided context to answer the question.
         if the answer is not present in the context,
         say:"I could not find the answer in the document."
         """),

         ("human",
          """
          Context:{context}
          Question:{question}
          """)
    ]
)

print("RAG")
print("press 0 to exit")

while True:
    query = input("You:")
    if query == "0":
        break
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI:{response.content}")
