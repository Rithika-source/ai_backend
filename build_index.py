# Run this ONCE to build and save the FAISS index
from rag.vector_store import build_vectorstore

with open("document.txt", "r") as f:
    text = f.read()

build_vectorstore(text)
print("\nIndex built! Now run:")
print("uvicorn api.main:app --reload") 