from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from core.config import FAISS_INDEX_PATH
import os

DOCS_FOLDER = "documents"
VECTORSTORE_PATH = FAISS_INDEX_PATH

def run_ingestion():
    # Step 1 — check documents folder exists
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        logger.warning(f"Created '{DOCS_FOLDER}' folder — add your files there and run again")
        return

    # Step 2 — load files
    docs = load_documents(DOCS_FOLDER)
    if not docs:
        logger.warning("No documents found in folder")
        return

    # Step 3 — split into chunks
    chunks = split_documents(docs)

    # Step 4 — generate embeddings + save to FAISS
    logger.info("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    logger.info(f"Saving to: {os.path.abspath(VECTORSTORE_PATH)}")
    logger.info(f"Vectorstore updated! {len(chunks)} chunks stored at '{VECTORSTORE_PATH}'")

if __name__ == "__main__":
    run_ingestion()