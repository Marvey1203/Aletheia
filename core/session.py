# core/session.py

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

from pydantic import BaseModel, Field
from .schemas import Trace  # Assuming Trace is in core/schemas.py

# --- Pydantic Models for Session Data ---

class Session(BaseModel):
    """
    Represents a single, continuous conversation with Aletheia.
    A session is a collection of cognitive traces.
    """
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: Optional[str] = None  # The name can be derived from the first query.
    traces: List[Trace] = Field(default_factory=list)

# --- The Session Manager ---

class SessionManager:
    """
    Manages all conversational sessions. It handles the creation, retrieval,
    and context generation for the AI's "short-term memory."
    """
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: Optional[str] = None
        self.create_new_session() # Start with a fresh session immediately.
        print("SessionManager initialized with a new, active session.")

    def create_new_session(self) -> Session:
        """
        Creates a new, empty session and sets it as the active one.
        """
        new_session = Session()
        self.sessions[new_session.session_id] = new_session
        self.current_session_id = new_session.session_id
        print(f"Created and activated new session: {self.current_session_id}")
        return new_session

    def get_current_session(self) -> Optional[Session]:
        """
        Retrieves the currently active session object.
        """
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None

    def add_trace_to_current_session(self, trace: Trace):
        """
        Adds a completed cognitive trace to the active session's history.
        Also names the session after the first query if it's unnamed.
        """
        session = self.get_current_session()
        if not session:
            print("Warning: No active session found. Creating a new one.")
            session = self.create_new_session()

        # If the session is new, name it after the first query.
        if not session.name:
            session.name = trace.query[:75] # Truncate for brevity

        session.traces.append(trace)
        print(f"Added trace {trace.trace_id} to session {session.session_id}")

    def generate_trifold_context(self, query: str, max_recent_traces: int = 3) -> str:
        """
        Generates the crucial "Trifold Context" string to be injected into the LLM's prompt.
        This creates the functionally infinite memory.

        V1 Implementation:
        - Summary: Concatenates the first and last answers.
        - Recent History: Formats the last N turns of the conversation.
        - New Query: The user's latest input.
        """
        session = self.get_current_session()
        if not session or not session.traces:
            return f"The user's query is: \"{query}\""

        # 1. High-Level Summary
        summary_intro = f"This conversation started with the topic: \"{session.traces[0].summary.answer}\""
        summary_conclusion = ""
        if len(session.traces) > 1:
            summary_conclusion = f" The most recent conclusion was: \"{session.traces[-1].summary.answer}\""
        
        high_level_summary = summary_intro + summary_conclusion

        # 2. Rolling Recent History
        recent_history_str = ""
        recent_traces = session.traces[-max_recent_traces:]
        if recent_traces:
            history_lines = []
            for trace in recent_traces:
                history_lines.append(f"Previously, User asked: \"{trace.query}\"")
                history_lines.append(f"Aletheia answered: \"{trace.summary.answer}\"")
            recent_history_str = "\n".join(history_lines)

        # 3. Assemble the final context string
        context = f"""--- Conversation Context ---
[Summary]: {high_level_summary}

[Recent History]:
{recent_history_str}

--- Current Task ---
The user's new query is: "{query}"
"""
        return context