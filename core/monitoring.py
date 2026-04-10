import time
import uuid
from loguru import logger

def estimate_tokens(text: str) -> int:
    # rough estimate: 1 token ≈ 4 characters
    return len(text) // 4

def log_request(query: str, answer: str, retrieval_time: float = 0.0, llm_time: float = 0.0):
    request_id = str(uuid.uuid4())[:8]  # short unique ID
    tokens = estimate_tokens(query + answer)

    logger.info(
        f"\n. REQUEST TRACE"
        f"\n  request_id    : {request_id}"
        f"\n  query         : {query[:80]}"  # trim long queries
        f"\n  retrieval_time: {round(retrieval_time, 3)}s"
        f"\n  llm_time      : {round(llm_time, 3)}s"
        f"\n  total_time    : {round(retrieval_time + llm_time, 3)}s"
        f"\n  est. tokens   : {tokens}"
    )

    return request_id