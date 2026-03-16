from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager

import uvicorn
import logging
import json
import time
import re
import base64
import asyncio

from concurrent.futures import ThreadPoolExecutor

import edge_tts

from app.models import ChatRequest, ChatResponse, TTSRequest

RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more. "
    "Please try again later."
)


def _is_rate_limit_error(exc: Exception) -> bool:

    msg = str(exc).lower()

    return (
        "429" in str(exc)
        or "rate limit" in msg
        or "tokens per day" in msg
    )


from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService, AllGroqApisFailedError
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService


from config import (
    VECTOR_STORE_DIR,
    GROQ_API_KEYS,
    GROQ_MODEL,
    TAVILY_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHAT_HISTORY_TURNS,
    ASSISTANT_NAME,
    TTS_VOICE,
    TTS_RATE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("J.A.R.V.I.S")


vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatService = None

def print_title():

    title = """

╔══════════════════════════════════════════════════════╗
║                                                      ║
║      J A R V I S                                     ║
║                                                      ║
║      Just A Rather Very Intelligent System           ║
║                                                      ║
╚══════════════════════════════════════════════════════╝

"""

    print(title)


@asynccontextmanager
async def lifespan(app: FastAPI):

    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()

    logger.info("=" * 60)
    logger.info("J.A.R.V.I.S - Starting Up...")
    logger.info("=" * 60)

    logger.info("[CONFIG] Assistant name: %s", ASSISTANT_NAME)
    logger.info("[CONFIG] Groq model: %s", GROQ_MODEL)
    logger.info("[CONFIG] Groq API keys loaded: %d", len(GROQ_API_KEYS))
    logger.info(
        "[CONFIG] Tavily API key: %s",
        "configured" if TAVILY_API_KEY else "NOT SET",
    )
    logger.info("[CONFIG] Embedding model: %s", EMBEDDING_MODEL)
    logger.info(
        "[CONFIG] Chunk size: %d | Overlap: %d | Max history turns: %d",
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MAX_CHAT_HISTORY_TURNS,
    )

    try:

        logger.info("Initializing vector store service...")
        t0 = time.perf_counter()

        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()

        logger.info(
            "[TIMING] startup_vector_store: %.3fs",
            time.perf_counter() - t0,
        )

        logger.info("Initializing Groq service (general queries)...")

        groq_service = GroqService(vector_store_service)

        logger.info("Groq service initialized successfully")

        logger.info(
            "Initializing Realtime Groq service (with Tavily search)..."
        )

        realtime_service = RealtimeGroqService(vector_store_service)

        logger.info("Realtime Groq service initialized successfully")

        logger.info("Initializing chat service...")

        chat_service = ChatService(
            groq_service,
            realtime_service,
        )

        logger.info("Chat service initialized successfully")

        logger.info("=" * 60)
        logger.info("Service Status:")
        logger.info(" - Vector Store: Ready")
        logger.info(" - Groq AI (General): Ready")
        logger.info(" - Groq AI (Realtime): Ready")
        logger.info(" - Chat Service: Ready")
        logger.info("=" * 60)

        logger.info("J.A.R.V.I.S is online and ready!")
        logger.info("API: http://localhost:8000")
        logger.info("Frontend: http://localhost:8000/app")

        logger.info("=" * 60)

        yield

        logger.info("Shutting down J.A.R.V.I.S...")

        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)

        logger.info("All sessions saved. Goodbye!")

    except Exception as e:

        logger.error(
            f"Fatal error during startup: {e}",
            exc_info=True,
        )

        raise

app = FastAPI(
    title="J.A.R.V.I.S API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TimingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        t0 = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - t0

        path = request.url.path

        logger.info(
            "[REQUEST] %s %s -> %s (%.3fs)",
            request.method,
            path,
            response.status_code,
            elapsed,
        )

        return response


app.add_middleware(TimingMiddleware)


@app.get("/api")
async def api_info():

    return {
        "message": "J.A.R.V.I.S API",
        "endpoints": {
            "/chat": "General chat (non-streaming)",
            "/chat/stream": "General chat (streaming chunks)",
            "/chat/realtime": "Realtime chat (non-streaming)",
            "/chat/realtime/stream": "Realtime chat (streaming)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System health",
            "/tts": "Text to speech",
        },
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "vector_store": vector_store_service is not None,
        "chat_service": chat_service is not None,
    }

app = FastAPI(
    title="J.A.R.V.I.S API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TimingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        t0 = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - t0

        path = request.url.path

        logger.info(
            "[REQUEST] %s %s -> %s (%.3fs)",
            request.method,
            path,
            response.status_code,
            elapsed,
        )

        return response


app.add_middleware(TimingMiddleware)


@app.get("/api")
async def api_info():

    return {
        "message": "J.A.R.V.I.S API",
        "endpoints": {
            "/chat": "General chat (non-streaming)",
            "/chat/stream": "General chat (streaming chunks)",
            "/chat/realtime": "Realtime chat (non-streaming)",
            "/chat/realtime/stream": "Realtime chat (streaming)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System health",
            "/tts": "Text to speech",
        },
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "vector_store": vector_store_service is not None,
        "chat_service": chat_service is not None,
    }

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS_FIRST = 2
_MIN_WORDS = 3
_MERGE_IF_WORDS = 2


def _split_sentences(buf: str):

    parts = _SPLIT_RE.split(buf)

    if len(parts) <= 1:
        return [], buf

    raw = [p.strip() for p in parts[:-1] if p.strip()]

    sentences = []
    pending = ""

    for s in raw:

        if pending:
            s = (pending + " " + s).strip()
            pending = ""

        min_req = (
            _MIN_WORDS_FIRST
            if not sentences
            else _MIN_WORDS
        )

        if len(s.split()) < min_req:
            pending = s
            continue

        sentences.append(s)

    remaining = (
        (pending + " " + parts[-1].strip()).strip()
        if pending
        else parts[-1].strip()
    )

    return sentences, remaining


def _merge_short(sentences):

    if not sentences:
        return []

    merged = []
    i = 0

    while i < len(sentences):

        cur = sentences[i]
        j = i + 1

        while (
            j < len(sentences)
            and len(sentences[j].split()) <= _MERGE_IF_WORDS
        ):
            cur = (cur + " " + sentences[j]).strip()
            j += 1

        merged.append(cur)
        i = j

    return merged

def _generate_tts_sync(text: str) -> bytes:

    async def _run():

        communicate = edge_tts.Communicate(
            text,
            TTS_VOICE,
            rate=TTS_RATE,
        )

        audio_bytes = b""

        async for chunk in communicate.stream():

            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        return audio_bytes

    return asyncio.run(_run())


tts_executor = ThreadPoolExecutor(max_workers=2)

def _stream_generator(
    session_id: str,
    message: str,
    realtime: bool,
    want_tts: bool,
):

    start = time.perf_counter()

    if realtime:
        stream_iter = chat_service.process_realtime_message_stream(
            session_id,
            message,
        )
    else:
        stream_iter = chat_service.process_message_stream(
            session_id,
            message,
        )

    buffer = ""
    first_chunk = True

    for chunk in stream_iter:

        if isinstance(chunk, dict):
            yield f"data: {json.dumps(chunk)}\n\n"
            continue

        buffer += chunk

        sentences, buffer = _split_sentences(buffer)

        sentences = _merge_short(sentences)

        for s in sentences:

            payload = {
                "chunk": s,
            }

            if want_tts:

                try:

                    audio = tts_executor.submit(
                        _generate_tts_sync,
                        s,
                    ).result()

                    payload["audio"] = base64.b64encode(
                        audio
                    ).decode()

                except Exception as e:

                    logger.warning(
                        "TTS failed: %s",
                        e,
                    )

            yield f"data: {json.dumps(payload)}\n\n"

        if first_chunk:
            logger.info(
                "[TIMING] first chunk: %.3fs",
                time.perf_counter() - start,
            )
            first_chunk = False

    if buffer.strip():

        payload = {
            "chunk": buffer.strip(),
        }

        if want_tts:

            try:

                audio = tts_executor.submit(
                    _generate_tts_sync,
                    buffer.strip(),
                ).result()

                payload["audio"] = base64.b64encode(
                    audio
                ).decode()

            except Exception as e:

                logger.warning(
                    "TTS failed: %s",
                    e,
                )

        yield f"data: {json.dumps(payload)}\n\n"

    yield "data: " + __import__("json").dumps({"chunk": "", "done": True}) + "\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    session_id = chat_service.get_or_create_session(
        request.session_id
    )

    generator = _stream_generator(
        session_id,
        request.message,
        realtime=False,
        want_tts=request.tts,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
    )


@app.post("/chat/realtime")
async def chat_realtime(request: ChatRequest):

    session_id = chat_service.get_or_create_session(
        request.session_id
    )

    response = chat_service.process_realtime_message_stream(
        session_id,
        request.message,
    )

    chat_service.save_chat_session(session_id)

    return ChatResponse(
        response=response,
        session_id=session_id,
    )


@app.post("/chat/realtime/stream")
async def chat_realtime_stream(request: ChatRequest):

    session_id = chat_service.get_or_create_session(
        request.session_id
    )

    generator = _stream_generator(
        session_id,
        request.message,
        realtime=True,
        want_tts=request.tts,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
    )


@app.get("/chat/history/{session_id}")
async def get_history(session_id: str):

    history = chat_service.get_chat_history(
        session_id
    )

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
            }
            for m in history
        ],
    }


@app.post("/tts")
async def tts(request: TTSRequest):

    try:

        audio = _generate_tts_sync(
            request.text
        )

        return {
            "audio": base64.b64encode(
                audio
            ).decode()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


app.mount(
    "/app",
    StaticFiles(
        directory=str(Path("frontend")),
        html=True,
    ),
    name="frontend",
)


@app.get("/")
async def root():

    return RedirectResponse(
        url="/app"
    )


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
