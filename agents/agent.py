from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from agents.tools import tools
from loguru import logger

# Load TinyLlama
def load_llm():
    pipe = hf_pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.1,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# Build a simple tool description string for the prompt
def get_tools_description():
    desc = ""
    for tool in tools:
        desc += f"- {tool.name}: {tool.description}\n"
    return desc

# Simple prompt that tells LLM which tools exist and asks it to pick one
AGENT_PROMPT = """You are a helpful assistant. You have access to these tools:

{tools}

To answer the question, decide which tool to use, call it, and return the answer.

Question: {question}

Think step by step:
1. Which tool is best for this question?
2. What input should I give it?
3. What is the final answer?

Answer:"""

def parse_math(question: str) -> str:
    import re
    q = question.lower()

    # square root — handle separately before anything else
    if "square root" in q:
        nums = re.findall(r"\d+\.?\d*", q)
        if nums:
            expr = f"{nums[0]} ** 0.5"
            logger.info(f"Parsed math expression: '{expr}'")
            return expr

    # percentage — handle separately
    if "percent" in q or "%" in q:
        nums = re.findall(r"\d+\.?\d*", q)
        if len(nums) >= 2:
            expr = f"{nums[0]} / 100 * {nums[1]}"
            logger.info(f"Parsed math expression: '{expr}'")
            return expr

    # extract numbers in order they appear
    nums = re.findall(r"\d+\.?\d*", q)

    if len(nums) < 2:
        return nums[0] if nums else q

    a, b = nums[0], nums[1]

    # pick operator based on keywords
    if any(t in q for t in ["product of", "multiplied by", "times"]):
        expr = f"{a} * {b}"
    elif any(t in q for t in ["divided by", "divide"]):
        expr = f"{a} / {b}"
    elif any(t in q for t in ["sum of", "plus", "added to"]):
        expr = f"{a} + {b}"
    elif any(t in q for t in ["difference between", "minus", "subtract"]):
        expr = f"{a} - {b}"
    else:
        # fallback for direct expressions like "10 + 5"
        expr = re.sub(r"[^0-9+\-*/().\s]", "", q).strip()

    logger.info(f"Parsed math expression: '{expr}'")
    return expr

def run_agent(question: str) -> str:
    logger.info(f"Agent received: {question}")
    try:
        q = question.lower().strip()

        # --- Tool selection by keywords (reliable, no LLM needed) ---

        math_triggers = [
            "divided by", "multiplied by", "times", "plus", "minus",
            "added to", "subtract", "multiply", "divide", "calculate",
            "what is", "sum of", "product of", "difference between",
            "square root", "percent", "%", "+", "-", "*", "/"
        ]

        search_triggers = ["latest", "news", "recent", "current", "today", "who is", "when did"]

        is_math = any(t in q for t in math_triggers) and any(c.isdigit() for c in q)
        is_search = any(t in q for t in search_triggers)

        if is_math:
            logger.info("Agent picked: Calculator")
            result = tools[0].func(parse_math(question))

        elif is_search:
            logger.info("Agent picked: WebSearch")
            result = tools[1].func(question)

        else:
            logger.info("Agent picked: DocumentSearch")
            result = tools[2].func(question)

        logger.info(f"Agent result: {result}")
        return result

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"Agent could not answer: {str(e)}"