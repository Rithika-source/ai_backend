from langchain_core.tools import Tool
from rag.retriever import answer_question

# Tool 1 — Calculator
def calculator(expression: str) -> str:
    """Use this for any math calculation. Input must be a math expression like '144 / 12'."""
    try:
        result = eval(expression)  # safe for simple math
        return f"The answer is {result}"
    except Exception as e:
        return f"Calculator error: {e}"

# Tool 2 — Web Search (we'll simulate it since TinyLlama is local)
def fake_web_search(query: str) -> str:
    """Use this to search for recent news or facts not in the documents."""
    return f"Web search result for '{query}': This is a simulated result. In production, connect to a real search API like SerpAPI."

# Tool 3 — Document RAG
def doc_search(question: str) -> str:
    """Use this to answer questions from the uploaded documents."""
    return answer_question(question)

# Register all tools
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations. Input should be a math expression like '10 * 5' or '144 / 12'."
    ),
    Tool(
        name="WebSearch",
        func=fake_web_search,
        description="Useful for searching recent news or general facts not found in documents."
    ),
    Tool(
        name="DocumentSearch",
        func=doc_search,
        description="Useful for answering questions from the uploaded documents using RAG."
    ),
]