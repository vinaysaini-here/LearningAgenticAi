# from dotenv import load_dotenv

# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore

# load_dotenv()

# pdf_path = Path(__file__).parent / "nodejs.pdf"

# # Load this file in python program
# loader = PyPDFLoader(file_path=pdf_path)
# docs = loader.load()

# # Split the docs into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=400
# )

# chunks = text_splitter.split_documents(documents=docs)

# # Vector Embeddings
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-large"
# )

# vector_store = QdrantVectorStore.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     url="http://localhost:6333",
#     collection_name="learning_rag"
# )

# print("Indexing of documents done....")





# Code working with gemini embeddings



from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load env variables
load_dotenv()

# PDF path
pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(docs)


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Store in Qdrant
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("✅ Indexing of documents done using Gemini embeddings")
