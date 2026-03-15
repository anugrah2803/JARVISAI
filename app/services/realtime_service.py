from typing import List, Optional
from tavily import TavilyClient
import logging
import os

from app.services.groq_service import GroqService, escape_curly_braces
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import JARVIS_SYSTEM_PROMPT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import GROQ_API_KEY, GROQ_MODEL, JARVIS_SYSTEM_PROMPT


logger = logging.getLogger("J.A.R.V.I.S")


class RealTimeGroqService(GroqService):

    def __init__(self, vector_store_service: VectorStoreService):
        super().__init__(vector_store_service)

        Tavily_api_key = os.getenv("TAVILY_API_KEY", "")

        if Tavily_api_key:
            self.tavily_client = TavilyClient(api_key=Tavily_api_key)
            logger.info("Tavily client initialized successfully.")
        else:
            self.tavily_client = None
            logger.warning("Tavily API key not found.")

    def search_tavily(self, query: str, num_results: int = 5) -> str:

        if not self.tavily_client:
            logger.error("Tavily client is not initialized.")
            return ""

        formatted_result = ""

        try:
            response = with_retry(
                lambda: self.tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=num_results,
                    include_answers=False,
                    include_raw_content=False,
                ),
                max_retries=3,
                initial_delay=1.0,
            )

            results = response.get("results", [])

            for i, result in enumerate(results[:num_results], 1):

                title = result.get("title", "No Title")
                content = result.get("content", "No Content")
                url = result.get("url", "")

                formatted_result += f"title: {title}\n"
                formatted_result += f"description: {content}\n"

                if url:
                    formatted_result += f"url: {url}\n"

                formatted_result += "\n"

            logger.info(
                f"Tavily search completed for query: {query} ({len(results)} results)"
            )

            return formatted_result

        except Exception as e:
            logger.error(f"Error during Tavily search: {e}")
            return ""

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> str:

        try:

            logger.info(f"Generating response for question: {question}")

            search_results = self.search_tavily(question, num_results=5)

            context = ""
            context_docs = []

            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_docs = retriever.invoke(question)

                context = (
                    "\n".join(
                        [doc.page_content for doc in context_docs]
                    )
                    if context_docs
                    else ""
                )

            except Exception as retrieve_err:
                logger.warning(
                    f"Error retrieving context from vector store: {retrieve_err}"
                )

            time_info = get_time_information()

            system_message = (
                JARVIS_SYSTEM_PROMPT
                + f"\nCurrent time and date: {time_info}"
            )

            if search_results:
                escaped_search_results = escape_curly_braces(
                    search_results
                )

                system_message += (
                    f"\n\nRelevant information from the web:\n"
                    f"{escaped_search_results}"
                )

            if context:
                escaped_context = escape_curly_braces(context)

                system_message += (
                    f"\n\nRelevant information from the vector store:\n"
                    f"{escaped_context}"
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
                for human_msg, ai_msg in chat_history:
                    messages.append(
                        HumanMessage(content=human_msg)
                    )
                    messages.append(
                        AIMessage(content=ai_msg)
                    )

            response_content = self._invoke_llm(
                prompt,
                messages,
                question,
            )

            logger.info(
                f"RealTime Response generated successfully for question: {question}"
            )

            return response_content

        except Exception as e:
            logger.error(
                f"Error generating response: {e}",
                exc_info=True,
            )
            raise