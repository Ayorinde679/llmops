from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# IMPORTANT: These are assumed to be your local library imports
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(title="MultiDocChat", version="0.1.0")

# CORS (optional for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates setup
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


# ----------------------------
# Simple in-memory chat history (Note: Consider using a proper caching service like Redis for production)
# ----------------------------
SESSIONS: Dict[str, List[Dict[str, str]]] = {}


# ----------------------------
# Adapters
# ----------------------------
class DocumentAdapter:
    """
    Adapts file content (bytes) into a simple object structure 
    that ChatIngestor's library expects (synchronous .getbuffer() and .name).
    """
    def __init__(self, filename: str, content: bytes):
        self.name = filename
        self._content = content

    def getbuffer(self) -> bytes:
        """Returns the full file content as bytes."""
        return self._content


# ----------------------------
# Models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    """
    ChatRequest expects session_id and message field.
    """
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # ASYNC FIX: Read all files asynchronously and adapt the data
        adapted_documents = []
        for f in files:
            # Await the file read to avoid blocking I/O and handle memory correctly
            content = await f.read() 
            adapted_documents.append(DocumentAdapter(filename=f.filename, content=content))

        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id

        # Save, load, split, embed, and write FAISS index with MMR
        ingestor.built_retriver(
            uploaded_files=adapted_documents,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Initialize empty history for this session
        SESSIONS[session_id] = []

        return UploadResponse(session_id=session_id, indexed=True, message="Indexing complete with MMR")
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Log the error for better debugging
        print(f"Upload Critical Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed due to an unexpected server error.")


@app.post("/query", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    question = req.message.strip()
    
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Build RAG and load retriever from persisted FAISS with MMR
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"faiss_index/{session_id}"
        rag.load_retriever_from_faiss(
            index_path=index_path,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Use simple in-memory history and convert to BaseMessage list
        simple = SESSIONS.get(session_id, [])
        lc_history = []
        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        # Invoke RAG with the current question and history
        answer = rag.invoke(question, chat_history=lc_history)

        # Update history
        simple.append({"role": "user", "content": question})
        simple.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = simple

        return ChatResponse(answer=answer)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Log the error for better debugging
        print(f"Chat Critical Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed due to an unexpected server error.")


# Uvicorn entrypoint for `python main.py` (optional)
if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 for containerized environments
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)