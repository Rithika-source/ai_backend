from langchain_community.embeddings import HuggingFaceEmbeddings
from core.config import EMBEDDING_MODEL

def get_embedder():
    # Converts text into a list of 384 numbers (a vector)
    # Similar meaning = similar numbers = close in vector space
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)