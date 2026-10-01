"""
PhytoSpectra-VLM / WheatSpectraChat
In-memory session and inference caching with TTL cleanup.
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from app.config import settings


class SessionCache:
    def __init__(self, ttl_seconds: int = settings.CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._images: Dict[str, Dict[str, Any]] = {}
        self.image_expiry_callback = None

    def create_session(self) -> str:
        self.cleanup_expired()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "created_at": time.time(),
            "last_accessed": time.time(),
            "chat_history": [],
            "image_ids": [],
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.cleanup_expired()
        session = self._sessions.get(session_id)
        if session:
            session["last_accessed"] = time.time()
        return session

    def store_image_data(
        self,
        session_id: str,
        image_id: str,
        data: Dict[str, Any],
    ) -> None:
        self.cleanup_expired()
        if session_id in self._sessions:
            self._sessions[session_id]["last_accessed"] = time.time()
            if image_id not in self._sessions[session_id]["image_ids"]:
                self._sessions[session_id]["image_ids"].append(image_id)

        data["session_id"] = session_id
        data["created_at"] = time.time()
        data["last_accessed"] = time.time()
        self._images[image_id] = data

    def get_image_data(self, image_id: str) -> Optional[Dict[str, Any]]:
        self.cleanup_expired()
        data = self._images.get(image_id)
        if data:
            data["last_accessed"] = time.time()
        return data

    def get_chat_history(self, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return []
        history = session.get("chat_history", [])
        return list(history[-limit:])

    def append_chat_turn(
        self,
        session_id: str,
        question: str,
        plan: Dict[str, Any],
        answer_type: str,
    ) -> None:
        session = self.get_session(session_id)
        if not session:
            return
        history = session.setdefault("chat_history", [])
        history.append(
            {
                "question": question,
                "plan": plan,
                "answer_type": answer_type,
            }
        )
        # Bound session memory; only recent turns are needed for reference resolution.
        if len(history) > 20:
            del history[:-20]

    def cleanup_expired(self) -> None:
        now = time.time()
        expired_sessions = [
            sid
            for sid, s in self._sessions.items()
            if now - s.get("last_accessed", 0) > self.ttl_seconds
        ]
        for sid in expired_sessions:
            del self._sessions[sid]

        expired_images = [
            img_id
            for img_id, img in self._images.items()
            if now - img.get("last_accessed", 0) > self.ttl_seconds
        ]
        for img_id in expired_images:
            del self._images[img_id]
            if self.image_expiry_callback is not None:
                self.image_expiry_callback(img_id)


session_cache = SessionCache()
