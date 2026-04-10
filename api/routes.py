from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
from rag.pipeline import chat_answer, stream_answer
from agents.agent import run_agent
from rag.retriever import retrieve_context, build_prompt, answer_question, get_llm
from rag.vector_store import load_vectorstore
from core.cache import get_cached, set_cache, clear_cache
from core.monitoring import log_request
import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Depends, Request
from core.security import verify_api_key

limiter = Limiter(key_func=get_remote_address)


# --- Standard response format ---
def success_response(data: dict):
    return {"status": "success", "data": data, "error": None}

def error_response(message: str):
    return {"status": "error", "data": None, "error": message}

router = APIRouter()

# --- Request body models ---
class AskRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class AgentRequest(BaseModel):
    question: str

class DebugRequest(BaseModel):
    question: str
    k: int = 3

# --- Endpoints ---

@router.post("/ask", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def ask(request: Request,req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        logger.info(f"/ask | question: {req.question}")

        # check cache first
        cached = get_cached(req.question)
        if cached:
            logger.info("/ask | returning cached answer")
            return success_response({"question": req.question, "answer": cached, "cached": True})

        # not cached — run RAG
        t1=time.time()
        answer = answer_question(req.question)
        llm_time=time.time()-t1

        log_request(query=req.question, answer= answer,llm_time=llm_time)

        # save to cache for next time
        set_cache(req.question, answer)

        return success_response({"question": req.question, "answer": answer, "cached": False})

    except Exception as e:
        logger.error(f"/ask | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def chat(request: Request,req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        logger.info(f"/chat | session: {req.session_id} | message: {req.message}")
        result = chat_answer(req.session_id, req.message)
        logger.info(f"/chat | session: {req.session_id} | response ready")
        return success_response(result)
    except Exception as e:
        logger.error(f"/chat | session: {req.session_id} | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        logger.info(f"/chat/stream | session: {req.session_id} | message: {req.message}")
        return StreamingResponse(
            stream_answer(req.session_id, req.message),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"/chat/stream | session: {req.session_id} | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))"""
    
@router.post("/agent", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def agent_ask(request: Request,req: AgentRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        logger.info(f"/agent | question: {req.question}")
        result = run_agent(req.question)
        return success_response({"question": req.question, "answer": result})
    except Exception as e:
        logger.error(f"/agent | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug-rag")
def debug_rag(req: DebugRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        logger.info(f"/debug-rag | question: {req.question}")

        # Step 1 — retrieve chunks from FAISS
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(req.question, k=req.k)

        # Step 2 — build the chunks list with metadata
        chunks = []
        for i, doc in enumerate(docs):
            chunks.append({
                "chunk_index": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            })

        # Step 3 — build context and get answer
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = build_prompt(context, req.question)
        llm = get_llm()
        answer = llm.invoke(prompt)

        logger.info(f"/debug-rag | {len(chunks)} chunks retrieved")

        return success_response({
            "question": req.question,
            "retrieved_chunks": chunks,
            "final_answer": answer,
            "total_chunks_retrieved": len(chunks),
        })

    except Exception as e:
        logger.error(f"/debug-rag | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))    

@router.post("/cache/clear")
def clear():
    clear_cache()
    return success_response({"message": "Cache cleared"})

   