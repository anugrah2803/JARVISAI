import json

# ============================================================
# FIX 1: main.py - "text" key ko "chunk" se replace karo
# FIX 2: main.py - [DONE] ko proper done event se replace karo
# FIX 3: chat_service.py - sahi methods aur class name fix
# ============================================================

# --- Fix main.py ---
with open("app/main.py", "r", encoding="utf-8") as f:
    main_content = f.read()

main_content = main_content.replace('"text": s,', '"chunk": s,')
main_content = main_content.replace('"text": buffer.strip(),', '"chunk": buffer.strip(),')
main_content = main_content.replace('yield "data: [DONE]\\n\\n"', 'yield "data: " + json.dumps({"chunk": "", "done": True}) + "\\n\\n"')

with open("app/main.py", "w", encoding="utf-8") as f:
    f.write(main_content)

print("main.py fixed")

# --- Fix chat_service.py - rewrite completely ---
chat_service_content = '''import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService

logger = logging.getLogger("J.A.R.V.I.S")

class ChatService:
    def __init__(self, Groq_Service: GroqService, RealTime_Service: RealtimeGroqService = None):
        self.groq_service = Groq_Service
        self.realtime_groq_service = RealTime_Service
        self.sessions: Dict[str, List[ChatMessage]] = {}

    def load_session_from_disk(self, session_id: str) -> bool:
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        if not filepath.exists():
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dist = json.load(f)
                messages = [
                    ChatMessage(role=msg.get("role"), content=msg.get("content"))
                    for msg in chat_dist.get("messages", [])
                ]
                self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning("Failed to load session %s from disk: %s", session_id, e)
            return False

    def get_or_create_session(self, session_id: str) -> str:
        if not session_id:
            new_session_id = str(uuid.uuid4())
            self.sessions[new_session_id] = []
            return new_session_id
        if session_id in self.sessions:
            return session_id
        if self.load_session_from_disk(session_id):
            return session_id
        self.sessions[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role=role, content=content))

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        return self.sessions.get(session_id, [])

    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> List[tuple]:
        messages = self.get_chat_history(session_id)
        history = []
        messages_to_process = messages[:-1] if exclude_last and messages else messages
        i = 0
        while i < len(messages_to_process) - 1:
            user_msg = messages_to_process[i]
            ai_msg = messages_to_process[i + 1]
            if user_msg.role == "user" and ai_msg.role == "assistant":
                history.append((user_msg.content, ai_msg.content))
                i += 2
            else:
                i += 1
        if len(history) > MAX_CHAT_HISTORY_TURNS:
            history = history[-MAX_CHAT_HISTORY_TURNS:]
        return history

    def process_message(self, session_id: str, user_message: str) -> str:
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        response = self.groq_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        return response

    def process_realtime_message(self, session_id: str, user_message: str) -> str:
        if not self.realtime_groq_service:
            raise ValueError("Realtime is not initialised.")
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        response = self.realtime_groq_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        return response

    def process_message_stream(self, session_id: str, user_message: str, tts: bool = False):
        response = self.process_message(session_id, user_message)
        for char in response:
            yield char

    def process_realtime_message_stream(self, session_id: str, user_message: str, tts: bool = False):
        response = self.process_realtime_message(session_id, user_message)
        for char in response:
            yield char

    def save_chat_session(self, session_id: str):
        if session_id not in self.sessions or not self.sessions[session_id]:
            return
        messages = self.sessions[session_id]
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        chat_dist = {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chat_dist, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save chat session %s to disk: %s", session_id, e)
'''

with open("app/services/chat_service.py", "w", encoding="utf-8") as f:
    f.write(chat_service_content)

print("chat_service.py fixed")
print("All fixes applied successfully!")
