# core/session.py (V2 - Atlas-Aware)

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field
from .schemas import Trace
from .atlas import ConceptualAtlas # Import the Atlas

# --- Pydantic Models for Session Data ---
class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4()}")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: Optional[str] = None
    traces: List[Trace] = Field(default_factory=list)

# --- The Session Manager ---
class SessionManager:
    """
    Manages conversational sessions and generates rich, multi-layered context
    by querying the Conceptual Atlas for long-term memory.
    """
    def __init__(self, atlas: ConceptualAtlas):
        self.atlas = atlas # The manager now has a direct line to the long-term memory
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: Optional[str] = None
        self.create_new_session()
        print("SessionManager (Atlas-Aware) initialized.")

    def create_new_session(self) -> Session:
        new_session = Session()
        self.sessions[new_session.session_id] = new_session
        self.current_session_id = new_session.session_id
        return new_session

    def get_current_session(self) -> Optional[Session]:
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None

    def add_trace_to_current_session(self, trace: Trace):
        session = self.get_current_session()
        if not session:
            session = self.create_new_session()
        if not session.name:
            session.name = trace.query[:75]
        session.traces.append(trace)

    def generate_trifold_context(self, query: str, max_recent_traces: int = 2) -> str:
        """
        Generates the crucial "Trifold Context" string using the Conceptual Atlas.
        
        V2 Implementation:
        - Long-Term Memory: Fetches semantically relevant memories from the Atlas.
        - Short-Term Memory: Formats the last N turns of the conversation.
        - New Query: The user's latest input.
        """
        session = self.get_current_session()
        
        # 1. Long-Term Memory (from Atlas)
        relevant_memories = self.atlas.query_atlas(query_text=query, n_results=2)
        long_term_memory_str = "[No relevant long-term memories found]"
        if relevant_memories:
            memory_lines = ["--- Relevant Long-Term Memories ---"]
            for mem in relevant_memories:
                memory_lines.append(f"Previously, you discussed '{mem.get('query')}' and concluded: '{mem.get('answer')}'")
            long_term_memory_str = "\n".join(memory_lines)

        # 2. Short-Term Memory (from current session)
        short_term_memory_str = "[No recent conversation history in this session]"
        if session and session.traces:
            recent_traces = session.traces[-max_recent_traces:]
            history_lines = ["--- Recent Conversation History ---"]
            for trace in recent_traces:
                history_lines.append(f"User asked: \"{trace.query}\"")
                history_lines.append(f"Aletheia answered: \"{trace.summary.answer}\"")
            short_term_memory_str = "\n".join(history_lines)
            
        # 3. Assemble the final context string
        context = f"""You are Aletheia, a sovereign AI. Your primary goal is to provide answers based on your own, verified memories and understanding of your architecture.

            --- Relevant Long-Term Memories (Your Core Knowledge) ---
            {long_term_memory_str}

            --- Recent Conversation History ---
            {short_term_memory_str}

            --- Current Task ---
            Analyze the user's query below. Before answering, you MUST determine if the retrieved Long-Term Memories are relevant. If they are, you MUST synthesize them to form the core of your answer.

            User Query: "{query}"
            """
        return context