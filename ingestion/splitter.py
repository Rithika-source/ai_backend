from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks