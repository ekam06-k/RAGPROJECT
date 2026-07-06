from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Step 1: Load PDF
loader = PyPDFLoader("data/IOTBasics.pdf")
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Step 3: Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
   model="models/gemini-embedding-2-preview"
)

# Step 4: Store chunks in ChromaDB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\n✅ Vector Database Created Successfully!")