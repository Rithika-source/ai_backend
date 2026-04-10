from rag.vector_store import load_vectorstore
from core.prompts import RAG_SYSTEM_PROMPT
from core.config import MODEL_NAME, TEMPERATURE, MAX_TOKENS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
import os

def get_llm(temperature=None):
    temp = temperature if temperature is not None else TEMPERATURE

    return OllamaLLM(
        model="llama3",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    )

def retrieve_context(query: str, k: int = 3) -> str:
    # Load FAISS, find top-k most similar chunks to the query
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def build_prompt(context: str, query: str, history: str = "") -> str:
    history_section = f"\nConversation so far:\n{history}" if history else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{RAG_SYSTEM_PROMPT}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context:
{context}
{history_section}
Question: {query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

def answer_question(query: str, history: str = "", temperature: float = None) -> str:
    context = retrieve_context(query)
    prompt = build_prompt(context, query, history)
    llm = get_llm(temperature=temperature)
    return llm.invoke(prompt)