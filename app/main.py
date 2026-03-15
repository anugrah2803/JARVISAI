"""
J.A.R.V.I.S MAIN API
====================

This module defines the FastAPI application and all HTTP endpoints. It is
designed for single-user use: one person runs one server (e.g. python run.py)
and uses it as their personal J.A.R.V.I.S backend. Many people can each run
their own copy of this code on their own machine.

ENDPOINTS:
GET  /              - Returns API name and list of endpoints.
GET  /health        - Returns status of all services (for monitoring).
POST /chat          - General chat: pure LLM, no web search. Uses learning data
                      and past chats via vector-store retrieval only.
POST /chat/realtime - Realtime chat: runs a Tavily web search first, then
                      sends results + context to Groq. Same session as /chat.
GET  /chat/history/{id} - Returns all messages for a session (general + realtime).

SESSION:
Both /chat and /chat/realtime use the same session_id. If you omit session_id,
the server generates a UUID and returns it; send it back on the next request
to continue the conversation. Sessions are saved to disk and survive restarts.

STARTUP:
On startup, the lifespan function builds the vector store from learning_data/*.txt
and chats_data/*.json, then creates Groq, Realtime, and Chat services. On shutdown,
it saves all in-memory sessions to disk.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.models import ChatRequest, ChatResponse

RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more. "
    "Please try again later."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService
from app.services.realtime_service import RealTimeGroqService
from app.services.chat_service import ChatService
from config import VECTOR_STORE_DIR
from langchain_community.vectorstores import FAISS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("J.A.R.V.I.S")


# ------------------------------------------------------------
# GLOBAL SERVICE REFERENCES
# ------------------------------------------------------------

vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealTimeGroqService = None
chat_service: ChatService = None


def print_title():
    """Print the J.A.R.V.I.S ASCII art banner to the console when the server starts."""

    title = """

    ┌───────────────────────────────────────────────┐

        ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
        ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
        ██║███████║██████╔╝██║   ██║██║███████╗
   ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
   ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
    ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

        Just A Rather Very Intelligent System

    └───────────────────────────────────────────────┘

    """

    print(title)


# ------------------------------------------------------------
# LIFESPAN (STARTUP / SHUTDOWN)
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()

    logger.info("=" * 60)
    logger.info("J.A.R.V.I.S - Starting Up...")
    logger.info("=" * 60)

    try:

        # -------- Vector Store --------

        logger.info("Initializing vector store service...")
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()
        logger.info("Vector store initialized successfully")

        # -------- Groq --------

        logger.info("Initializing Groq service (general queries)...")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service initialized successfully")

        # -------- Realtime --------

        logger.info("Initializing Realtime Groq service (with Tavily search)...")
        realtime_service = RealTimeGroqService(vector_store_service)
        logger.info("Realtime Groq service initialized successfully")

        # -------- Chat --------

        logger.info("Initializing chat service...")
        chat_service = ChatService(groq_service, realtime_service)
        logger.info("Chat service initialized successfully")

        # -------- Done --------

        logger.info("=" * 60)
        logger.info("Service Status:")
        logger.info(" - Vector Store: Ready")
        logger.info(" - Groq AI (General): Ready")
        logger.info(" - Groq AI (Realtime): Ready")
        logger.info(" - Chat Service: Ready")
        logger.info("=" * 60)

        logger.info("J.A.R.V.I.S is online and ready!")
        logger.info("Docs: http://localhost:8000/docs")
        logger.info("=" * 60)

        yield

        # -------- Shutdown --------

        logger.info("Shutting down J.A.R.V.I.S...")

        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)

        logger.info("All sessions saved. Goodbye!")

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise

# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------

app = FastAPI(
    title="J.A.R.V.I.S API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------


@app.get("/")
async def root():

    return {
        "message": "J.A.R.V.I.S API",
        "endpoints": {
            "/chat": "General chat (pure LLM, no web search)",
            "/chat/realtime": "Realtime chat (with Tavily search)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System health check",
        },
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "vector_store": vector_store_service is not None,
        "groq_service": groq_service is not None,
        "realtime_service": realtime_service is not None,
        "chat_service": chat_service is not None,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not chat_service:
        raise HTTPException(
            status_code=503,
            detail="Chat service not initialized",
        )

    try:

        session_id = chat_service.get_or_create_session(
            request.session_id
        )

        response_text = chat_service.process_message(
            session_id,
            request.message,
        )

        chat_service.save_chat_session(session_id)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
        )

    except ValueError as e:

        logger.warning(f"Invalid session id: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        if _is_rate_limit_error(e):

            logger.warning(f"Rate limit hit: {e}")

            raise HTTPException(
                status_code=429,
                detail=RATE_LIMIT_MESSAGE,
            )

        logger.error(
            f"Error processing chat: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}",
        )


# ------------------------------------------------------------


@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):

    if not chat_service:
        raise HTTPException(
            status_code=503,
            detail="Chat service not initialized",
        )

    if not realtime_service:
        raise HTTPException(
            status_code=503,
            detail="Realtime service not initialized",
        )

    try:

        session_id = chat_service.get_or_create_session(
            request.session_id
        )

        response_text = chat_service.process_realtime_message(
            session_id,
            request.message,
        )

        chat_service.save_chat_session(session_id)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
        )

    except ValueError as e:

        logger.warning(f"Invalid session id: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        if _is_rate_limit_error(e):

            logger.warning(f"Rate limit hit: {e}")

            raise HTTPException(
                status_code=429,
                detail=RATE_LIMIT_MESSAGE,
            )

        logger.error(
            f"Error processing realtime chat: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}",
        )


# ------------------------------------------------------------


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):

    if not chat_service:
        raise HTTPException(
            status_code=503,
            detail="Chat service not initialized",
        )

    try:

        messages = chat_service.get_chat_history(
            session_id
        )

        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in messages
            ],
        }

    except Exception as e:

        logger.error(
            f"Error retrieving history: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}",
        )


# ------------------------------------------------------------
# STANDALONE RUN
# ------------------------------------------------------------


def run():

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run()