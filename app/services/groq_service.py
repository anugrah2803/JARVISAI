from typing import List, Optional, Iterator

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging
import time

from config import (
    GROQ_API_KEYS,
    GROQ_MODEL,
    JARVIS_SYSTEM_PROMPT,
    GENERAL_CHAT_ADDENDUM,
)

from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry


logger = logging.getLogger("J.A.R.V.I.S")


# ===============================
# CONSTANTS
# ===============================

GROQ_REQUEST_TIMEOUT = 60


ALL_APIS_FAILED_MESSAGE = (
    "I'm unable to process your request at the moment. "
    "All API services are temporarily unavailable. "
    "Please try again in a few minutes."
)


class AllGroqApisFailedError(Exception):
    """
    Raised when every configured Groq API key has been tried and all failed.
    """
    pass


# ===============================
# HELPER
# ===============================

def escape_curly_braces(text: str) -> str:
    """
    Double every { and } so LangChain does not treat them as template variables.
    """

    if not text:
        return text

    return text.replace("{", "{{").replace("}", "}}")
def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Check if error is rate limit / quota error.
    """

    msg = str(exc).lower()

    return (
        "429" in msg
        or "rate limit" in msg
        or "quota" in msg
        or "too many requests" in msg
        or "tokens per day" in msg
    )


def _log_timing(label: str, elapsed: float, extra: str = ""):
    """
    Log timing in consistent format.
    """

    msg = f"[TIMING] {label}: {elapsed:.3f}s"

    if extra:
        msg += f" ({extra})"

    logger.info(msg)


def _mask_api_key(key: str) -> str:
    """
    Show only first 8 and last 4 chars.
    """

    if not key or len(key) <= 12:
        return "***masked***"

    return f"{key[:8]}...{key[-4:]}"


# ===================================
# GROQ SERVICE
# ===================================


class GroqService:
    """
    General chat service using Groq + vector store.
    """
    def __init__(self, vector_store_service: VectorStoreService):

        if not GROQ_API_KEYS:
            raise ValueError(
                "No GROQ API keys configured. "
                "Set GROQ_API_KEY (and optionally GROQ_API_KEY_2, etc.) in .env"
            )

        # create one client per key
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.6,
                request_timeout=GROQ_REQUEST_TIMEOUT,
            )
            for key in GROQ_API_KEYS
        ]

        self.vector_store_service = vector_store_service

        logger.info(
            f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s)"
        )

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages,
        question: str,
    ) -> str:

        n = len(self.llms)

        last_exc = None

        for i in range(n):

            llm = self.llms[i]

            key = GROQ_API_KEYS[i]

            masked = _mask_api_key(key)

            start = time.perf_counter()

            try:

                chain = prompt | llm

                response = with_retry(
                    lambda: chain.invoke(
                        {
                            "history": messages,
                            "question": question,
                        }
                    )
                )

                elapsed = time.perf_counter() - start

                _log_timing(
                    "_invoke_llm",
                    elapsed,
                    f"api={i+1}/{n} {masked}",
                )

                return response.content

            except Exception as e:

                elapsed = time.perf_counter() - start

                last_exc = e

                if _is_rate_limit_error(e):

                    logger.warning(
                        f"API {i+1}/{n} rate limit: {masked}"
                    )

                else:

                    logger.error(
                        f"API {i+1}/{n} failed: {masked} {e}"
                    )

                continue

        raise AllGroqApisFailedError(
            ALL_APIS_FAILED_MESSAGE
        )

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> str:

        try:

            context = ""

            # --------------------------
            # VECTOR STORE
            # --------------------------

            try:

                retriever = self.vector_store_service.get_retriever(
                    k=10
                )

                context_docs = retriever.invoke(question)

                context = (
                    "\n".join(
                        doc.page_content
                        for doc in context_docs
                    )
                    if context_docs
                    else ""
                )

            except Exception as e:

                logger.warning(
                    "Vector store failed: %s",
                    e,
                )

            # --------------------------
            # TIME INFO
            # --------------------------

            time_info = get_time_information()

            system_message = (
                JARVIS_SYSTEM_PROMPT
                + "\n\n"
                + GENERAL_CHAT_ADDENDUM
                + f"\n\n{time_info}"
            )

            # --------------------------
            # ADD CONTEXT
            # --------------------------

            if context:

                system_message += (
                    "\n\nRelevant context from your data:\n"
                    + escape_curly_braces(context)
                )

            # --------------------------
            # PROMPT TEMPLATE
            # --------------------------

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )

            messages = []

            if chat_history:

                for human, ai in chat_history:

                    messages.append(
                        HumanMessage(content=human)
                    )

                    messages.append(
                        AIMessage(content=ai)
                    )

            result = self._invoke_llm(
                prompt,
                messages,
                question,
            )

            return result

        except AllGroqApisFailedError:

            raise

        except Exception as e:

            raise Exception(
                f"Error getting response from Groq: {e}"
            )

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> Iterator[str]:

        try:

            context = ""

            try:

                retriever = self.vector_store_service.get_retriever(
                    k=10
                )

                context_docs = retriever.invoke(question)

                context = (
                    "\n".join(
                        doc.page_content
                        for doc in context_docs
                    )
                    if context_docs
                    else ""
                )

            except Exception as e:

                logger.warning(
                    "Vector store failed: %s",
                    e,
                )

            time_info = get_time_information()

            system_message = (
                JARVIS_SYSTEM_PROMPT
                + "\n\n"
                + GENERAL_CHAT_ADDENDUM
                + f"\n\n{time_info}"
            )

            if context:

                system_message += (
                    "\n\nRelevant context:\n"
                    + escape_curly_braces(context)
                )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )

            messages = []

            if chat_history:

                for human, ai in chat_history:

                    messages.append(
                        HumanMessage(content=human)
                    )

                    messages.append(
                        AIMessage(content=ai)
                    )

            # stream using first API that works

            for i, llm in enumerate(self.llms):

                try:

                    chain = prompt | llm

                    stream = chain.stream(
                        {
                            "history": messages,
                            "question": question,
                        }
                    )

                    for chunk in stream:

                        yield chunk.content

                    return

                except Exception as e:

                    if _is_rate_limit_error(e):

                        logger.warning(
                            f"Rate limit API {i+1}"
                        )

                    else:

                        logger.error(e)

                    continue

            raise AllGroqApisFailedError(
                ALL_APIS_FAILED_MESSAGE
            )

        except Exception as e:

            raise Exception(
                f"Error streaming response: {e}"
            )
