# pipeline.py ties everything together as one clean callable
# This is what routes.py will import — keeps routes clean

from rag.retriever import answer_question, retrieve_context, build_prompt
from core.config import MODEL_NAME, MAX_TOKENS
from transformers import pipeline, TextIteratorStreamer
from threading import Thread

# In-memory chat history store
# { "session_id": [ {"role": "user", "content": "..."}, ... ] }
chat_sessions = {}

def get_history_text(session_id: str) -> str:
    if session_id not in chat_sessions:
        return ""
    messages = chat_sessions[session_id][-4:]  # last 2 turns (4 messages)
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

def save_message(session_id: str, role: str, content: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    chat_sessions[session_id].append({"role": role, "content": content})

def chat_answer(session_id: str, message: str) -> dict:
    # Get past conversation for this session
    history = get_history_text(session_id)

    # Save user message
    save_message(session_id, "user", message)

    # Get answer (history passed so LLM knows the conversation context)
    answer = answer_question(query=message, history=history)

    # Save assistant reply
    save_message(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "answer": answer,
        "turn": len(chat_sessions[session_id]) // 2
    }

def stream_answer(session_id: str, message: str):
    # Generator function — yields tokens one by one for StreamingResponse
    context = retrieve_context(message)
    history = get_history_text(session_id)
    prompt = build_prompt(context, message, history)

    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        max_new_tokens=MAX_TOKENS,
    )
    streamer = TextIteratorStreamer(
        pipe.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    inputs = pipe.tokenizer(prompt, return_tensors="pt")

    # Run generation in background thread so we can stream simultaneously
    thread = Thread(target=pipe.model.generate, kwargs={
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_TOKENS
    })
    thread.start()

    full_reply = ""
    for token in streamer:
        full_reply += token
        yield token  # sends each token to client as it's generated

    # Save complete reply to history after streaming finishes
    save_message(session_id, "user", message)
    save_message(session_id, "assistant", full_reply)