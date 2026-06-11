from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MedicalRAGSystem import MedicalRAGSystem

# ── RAG instance ──────────────────────────────────────────────────────────────
rag: MedicalRAGSystem | None = None


# ── Startup: initialize RAG automatically ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    try:
        rag = MedicalRAGSystem()
        rag.initialize_all()
        print("✅ RAG System initialized automatically on startup")
    except Exception as e:
        print(f"❌ Startup initialization failed: {str(e)}")
    yield
    # shutdown logic here if needed


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Medaura API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    top_k: int = 5
    use_history: bool = True


class Source(BaseModel):
    record_id: str
    urgency: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    total_records: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {"initialized": rag is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global rag
    if rag is None:
        return ChatResponse(
            answer="⚠️ System is still loading, please wait a moment and try again.",
            sources=[],
            total_records=0
        )
    try:
        if req.use_history:
            response = rag.ask_with_history(req.message, top_k=req.top_k)
        else:
            response = rag.ask(req.message, top_k=req.top_k)

        sources = [
            Source(
                record_id=doc.metadata.get("record_id", "?"),
                urgency=doc.metadata.get("urgency", "unknown")
            )
            for doc in response.sources
        ]

        return ChatResponse(
            answer=response.answer,
            sources=sources,
            total_records=response.total_records_found
        )
    except Exception as e:
        return ChatResponse(answer=f"❌ Error: {str(e)}", sources=[], total_records=0)


@app.post("/clear")
def clear():
    global rag
    if rag:
        rag.clear_history()
    return {"status": "ok", "message": "History cleared"}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
