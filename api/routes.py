from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
from rag.pipeline import chat_answer, stream_answer
from agents.agent import run_agent
from rag.retriever import retrieve_context, build_prompt, answer_question, get_llm
from rag.vector_store import load_vectorstore
from core.cache import get_cached, set_cache, clear_cache

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

@router.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        logger.info(f"/ask | question: {req.question}")

        # check cache first
        cached = get_cached(req.question)
        if cached:
            logger.info("/ask | returning cached answer")
            return {"question": req.question, "answer": cached, "cached": True}

        # not cached — run RAG
        answer = answer_question(req.question)

        # save to cache for next time
        set_cache(req.question, answer)

        return {"question": req.question, "answer": answer, "cached": False}

    except Exception as e:
        logger.error(f"/ask | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        logger.info(f"/chat | session: {req.session_id} | message: {req.message}")
        result = chat_answer(req.session_id, req.message)
        logger.info(f"/chat | session: {req.session_id} | response ready")
        return result
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
    
@router.post("/agent")
def agent_ask(req: AgentRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        logger.info(f"/agent | question: {req.question}")
        result = run_agent(req.question)
        return {"question": req.question, "answer": result}
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

        return {
            "question": req.question,
            "retrieved_chunks": chunks,
            "final_answer": answer,
            "total_chunks_retrieved": len(chunks),
        }

    except Exception as e:
        logger.error(f"/debug-rag | failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))    

@router.post("/cache/clear")
def clear():
    clear_cache()
    return {"message": "Cache cleared"}

   