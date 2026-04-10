import os
from loguru import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(folder_path: str) -> list:
    docs = []
    supported = (".pdf", ".txt", ".md")

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported:
            continue

        filepath = os.path.join(folder_path, filename)
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            else:
                loader = TextLoader(filepath, encoding="utf-8")

            loaded = loader.load()
            docs.extend(loaded)
            logger.info(f"Loaded: {filename} ({len(loaded)} chunks)")

        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs