from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
loader = PyPDFLoader("data/IOTBasics.pdf")
documents = loader.load()

# Step 2: Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

# Step 3: Split the document into chunks
chunks = text_splitter.split_documents(documents)

# Step 4: Print summary
print("=" * 70)
print(f"Total Pages  : {len(documents)}")
print(f"Total Chunks : {len(chunks)}")
print("=" * 70)

# Step 5: Print all chunks
for i, chunk in enumerate(chunks):

    print("\n" + "=" * 70)
    print(f"CHUNK NUMBER : {i + 1}")
    print("=" * 70)

    print("\nChunk Content:\n")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)

print("\n")
print("=" * 70)
print("All chunks printed successfully!")
print("=" * 70)