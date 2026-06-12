from contextlib import asynccontextmanager
import threading
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MedicalRAGSystem import MedicalRAGSystem

# ── Persistence location ──────────────────────────────────────────────────────
# Point DATA_DIR at a Railway Volume (e.g. /data) so the Chroma index survives
# restarts. Falls back to a local "db" folder so it still runs with no volume.
DATA_DIR = os.getenv("DATA_DIR", "db")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# ── RAG instance + readiness flags ────────────────────────────────────────────
rag: MedicalRAGSystem | None = None
ready: bool = False
init_error: str | None = None


def _initialize_rag() -> None:
    """Heavy startup work. Runs in a background thread so the web server can
    start accepting requests immediately and /status can report progress."""
    global rag, ready, init_error
    try:
        instance = MedicalRAGSystem(persist_directory=CHROMA_DIR)
        instance.initialize_all()
        rag = instance
        ready = True
        print("✅ RAG System initialized (background thread complete)")
    except Exception as e:
        init_error = str(e)
        print(f"❌ Startup initialization failed: {init_error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start init in the background and yield right away — do NOT block here,
    # otherwise the server won't accept connections until loading finishes.
    threading.Thread(target=_initialize_rag, daemon=True).start()
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

# NOTE: mounting the whole project dir would expose .env, source, and data over
# HTTP. Serve only a dedicated ./static folder if/when you need static assets.
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/favicon.ico")
def favicon():
    # Silence the harmless 404 the browser triggers for the tab icon.
    return Response(status_code=204)


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
    return {"initialized": ready, "error": init_error}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not ready or rag is None:
        msg = (
            f"❌ Initialization failed: {init_error}"
            if init_error
            else "⚠️ System is still loading, please wait a moment and try again."
        )
        return ChatResponse(answer=msg, sources=[], total_records=0)

    try:
        # NOTE: MedicalRAGSystem only exposes ask_with_history(); there is no
        # ask(). Both branches use it so use_history=False can't crash.
        response = rag.ask_with_history(req.message, top_k=req.top_k)

        sources = [
            Source(
                record_id=doc.metadata.get("record_id", "?"),
                urgency=doc.metadata.get("urgency", "unknown"),
            )
            for doc in response.sources
        ]

        return ChatResponse(
            answer=response.answer,
            sources=sources,
            total_records=response.total_records_found,
        )
    except Exception as e:
        return ChatResponse(answer=f"❌ Error: {str(e)}", sources=[], total_records=0)


@app.post("/clear")
def clear():
    if rag:
        rag.clear_history()
    return {"status": "ok", "message": "History cleared"}


# ── Run (local dev only; Railway uses the Procfile) ───────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
