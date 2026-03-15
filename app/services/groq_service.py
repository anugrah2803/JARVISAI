from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

from config import GROQ_API_KEY, GROQ_MODEL, JARVIS_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S")


def escape_curly_braces(text: str) -> str:
    
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "token per day" in msg


def _mask_api_key(key: str) -> str:
    if not key or len(key) < 12:
        return "****masked****"
    return f"{key[:8]}....{key[-4:]}"


class GroqService:
    _shared_key_index = 0
    _lock = None

    def __init__(self, vector_store_service: VectorStoreService):
        if not GROQ_API_KEY:
            raise ValueError(
                "No groq API key provided. Please set the GROQ_API_KEY environment variable."
            )
        
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.8,
            )
            for key in GROQ_API_KEY
        ]

        self.vector_store_service = vector_store_service
        logger.info(f"Initialized GroqService with {len(GROQ_API_KEY)} API key(s).")
    

    def _invoke_llm(
            self,
            prompt: ChatPromptTemplate,
            messages: List,
            question: str,
    ) -> str:
        
        n = len(self.llms)
        start_i = GroqService._shared_key_index % n
        current_key_index = GroqService._shared_key_index
        GroqService._shared_key_index += 1

        masked_keys = _mask_api_key(GROQ_API_KEY[start_i])
        logger.info(f"Using API key #{start_i + 1}/{n} (round-robin index: {current_key_index}): {masked_keys}")

        last_exc = None
        key_tried = []

        for j in range(n):
            i = (start_i + j) % n
            key_tried.append(i)
            try:
                chain = prompt | self.llms[i]
                response = chain.invoke({"history": messages, "question": question})

                if j > 0:
                    masked_success_key = _mask_api_key(GROQ_API_KEY[i])
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} succeeded: {masked_success_key}")

                return response.content

            except Exception as e:
                last_exc = e
                masked_failed_key = _mask_api_key(GROQ_API_KEY[i])

                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n}: rate limited: {masked_failed_key}")
                else:
                    logger.error(f"API key #{i + 1}/{n}: failed: {masked_failed_key} - {str(e)[:100]}")

                if n > 1:
                    continue

                raise Exception(f"Error geting responce from groq: {str(e)}") from e

        masked_all_keys = ", ".join([_mask_api_key(GROQ_API_KEY[i]) for i in key_tried])
        logger.error(f"All API keys tried and failed: {masked_all_keys}")

        raise Exception(f"Error getting response from groq: {str(last_exc)}") from last_exc
    

    def get_responce(
            self,
            question: str,
            chat_history: Optional[List[tuple]] = None
    ) -> str:
        try:

            context = ""

            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_docs = retriever.invoke(question)

                context = "\n".join(
                    [doc.page_content for doc in context_docs]
                ) if context_docs else ""

            except Exception as retrival_err:
                logger.warning(
                    "Vector store retrieval failed, using empty context: %s",
                    retrival_err
                )

            time_info = get_time_information()

            system_message = (
                JARVIS_SYSTEM_PROMPT
                + f"\n\nCurrent time and date: {time_info})"
            )

            if context:
                system_message += (
                    f"\n\nRelevant information retrieved from your memory:\n"
                    f"{escape_curly_braces(context)}"
                )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}")
                ]
            )

            messages = []

            if chat_history:
                for human_msg, ai_msg in chat_history:
                    messages.append(HumanMessage(content=human_msg))
                    messages.append(AIMessage(content=ai_msg))

            return self._invoke_llm(prompt, messages, question)

        except Exception as e:
            raise Exception(
                f"Error getting response from Groq: {str(e)}"
            ) from e