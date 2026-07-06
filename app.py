from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("data/IOTBasics.pdf") #tell location of pdf

# Read all pages
documents = loader.load()

# Print total number of pages
print(f"Total Pages: {len(documents)}")

# Print the first page
print("\n========== FIRST PAGE ==========\n")
print(documents[0].page_content)

# Print metadata
print("\n========== METADATA ==========")
print(documents[0].metadata)