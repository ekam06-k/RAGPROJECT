from dotenv import load_dotenv
import os

from google import genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API key
load_dotenv()

# Gemini Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Embedding model (must be the SAME model used in create_db.py)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview"
)

# Load Chroma Database
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("✅ Chroma Database Loaded Successfully!")

while True:

    question = input("\nAsk your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # Retrieve top 3 relevant chunks
    docs = db.similarity_search(question, k=3)

    print("\nRetrieved Chunks:\n")

    context = ""

    for i, doc in enumerate(docs):
        print("=" * 60)
        print(f"Chunk {i+1}")
        print(doc.page_content)
        print()

        context += doc.page_content + "\n\n"

    # Prompt Gemini
    prompt = f"""
You are an AI assistant.

Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\n" + "=" * 60)
    print("Gemini Answer\n")
    print(response.text)
    print("=" * 60)