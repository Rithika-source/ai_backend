------------
docker build --no-cache -t ai-backend . 
docker run -p 8000:8000 \-e OLLAMA_BASE_URL=http://host.docker.internal:11434 \ai-backend
------------
ollama serve
python3 -m uvicorn api.main:app --reload
------------
