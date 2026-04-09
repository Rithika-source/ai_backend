from rag.vector_store import load_vectorstore
from core.prompts import RAG_SYSTEM_PROMPT
from core.config import MODEL_NAME, TEMPERATURE, MAX_TOKENS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import Ollama

def get_llm(temperature=None):
    temp = temperature if temperature is not None else TEMPERATURE

    return Ollama(
        model="llama3",
        temperature=temp,
        base_url="http://localhost:11434"  # IMPORTANT
    )

def retrieve_context(query: str, k: int = 3) -> str:
    # Load FAISS, find top-k most similar chunks to the query
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def build_prompt(context: str, query: str, history: str = "") -> str:
    # TinyLlama uses this special format for system/user/assistant
    history_section = f"\nConversation so far:\n{history}" if history else ""
    return f"""<|system|>
{RAG_SYSTEM_PROMPT}
<|user|>
Context:
{context}
{history_section}
Question: {query}
<|assistant|>"""

def answer_question(query: str, history: str = "", temperature: float = None) -> str:
    context = retrieve_context(query)
    prompt = build_prompt(context, query, history)
    llm = get_llm(temperature=temperature)
    return llm.invoke(prompt)