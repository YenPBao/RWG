from rl_env.states import Policy 

from __future__ import annotations
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from src.model import load_model
from pydantic import BaseModel, Field
#Messaging primitives
@dataclass
class Message:
    sender: str
    topic: str
    content: Any
    receiver: Optional[str] = None           
    corr_id: Optional[str] = None            
    created_at: float = field(default_factory=lambda: time.time())
    meta: Dict[str, Any] = field(default_factory=dict)

    def fork(self, **overrides) -> "Message":
        d = {**self.__dict__, **overrides}
        return Message(**d)

# Blackboard (shared memory) 
class Blackboard:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._reducers: Dict[str, Callable[[Any, Any], Any]] = {} # funtion 

    def set(self, key: str, value: Any):
        self._data[key] = value

    def get(self, key: str, default=None) -> Any:
        return self._data.get(key, default)

    def register_reducer(self, key: str, reducer: Callable[[Any, Any], Any]):
        self._reducers[key] = reducer

    def merge(self, key: str, value: Any):
        if key in self._data and key in self._reducers:
            self._data[key] = self._reducers[key](self._data[key], value)
        else:
            self._data[key] = value


    def snapshot(self) -> Dict[str, Any]:
        return dict(self._data)

# Common reducers
def reducer_append_list(l: List[Any], r: List[Any]): return (l or []) + (r or [])
def reducer_set_max(a: float, b: float): return max(a, b)
def reducer_extend_set(a: Set[Any], b: Set[Any]): return set(a or set()).union(b or set())

# Message Bus (topic pub/sub)
class MessageBus:
    def __init__(self):
        self._topics: Dict[str, List[asyncio.Queue]] = {}
        self._direct_inbox: Dict[str, asyncio.Queue] = {}

    def register_agent_inbox(self, agent_id: str, inbox: asyncio.Queue):
        self._direct_inbox[agent_id] = inbox

    def subscribe(self, topic: str, inbox: asyncio.Queue):
        self._topics.setdefault(topic, []).append(inbox)

    async def publish(self, msg: Message):
        # Direct message
        if msg.receiver:
            q = self._direct_inbox.get(msg.receiver)
            if q: await q.put(msg); return
        # Topic broadcast
        for q in self._topics.get(msg.topic, []):
            await q.put(msg)
    
@dataclass
class AgentSpec:
    token_budget: int = 8000           
    context_capacity: int = 32000       
    cost_per_1k_tokens: float = 0.0   
    # tham số policy
    alpha: float = 1.0                  
    beta: float = 1.0                  
    epsilon: float = 0.05             
    confidence_threshold: float = 0.6   
    # tốc độ sinh văn bản (tokens/phút)
    gen_rate_hint: int = 180          
    # công cụ & tầm với
    toolset: List[str] = field(default_factory=lambda: ["search", "summarize"])
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class SessionLocal:
    invited_for: Optional[str] = None
    need_more_sources: bool = True
    kb_size: int = 0
    tokens_used: int = 0
    last_confidence: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)  # logs

class DebateFeedback(BaseModel):
    """Lưu trữ kết quả của MỘT vòng debate"""
    round: int
    score: float = 0.0
    reasoning: str = ""
    critiques: List[str] = Field(default_factory=list)

class DebateState(BaseModel):
    """
    Đây là Trạng thái chung (S) hay 'Blackboard'.
    """
    session_id: str
    prompt: str 
    
    outline: List[str] = Field(default_factory=list)
    draft: str = ""
    feedback_history: List[DebateFeedback] = Field(default_factory=list)
    
    round: int = 0
    max_rounds: int = 2