from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.embedder import get_embedder
from core.config import FAISS_INDEX_PATH

def build_vectorstore(text: str):
    # Split document into overlapping chunks
    # chunk_size=500 → each chunk is ~500 characters
    # chunk_overlap=50 → 50 characters shared between chunks
    #   (prevents losing meaning at chunk boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents([text])
    print(f"Split into {len(chunks)} chunks")

    # Embed all chunks and store in FAISS index
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)

    # Save to disk so you don't re-embed every run
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}/'")
    return vectorstore

def load_vectorstore():
    embedder = get_embedder()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embedder,
        allow_dangerous_deserialization=True  # needed because FAISS uses pickle internally
    )