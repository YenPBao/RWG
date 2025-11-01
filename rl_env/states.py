from rl_env.states import Policy 

from __future__ import annotations
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

#Messaging primitives
@dataclass
class Message:
    sender: str
    topic: str
    content: Any
    receiver: Optional[str] = None           # None => broadcast on topic
    corr_id: Optional[str] = None            # correlation for request/response
    created_at: float = field(default_factory=lambda: time.time())
    meta: Dict[str, Any] = field(default_factory=dict)

    def fork(self, **overrides) -> "Message":
        d = {**self.__dict__, **overrides}
        return Message(**d)

# Blackboard (shared memory) with reducers
class Blackboard:
    """
    Simple KV store with merge reducers per key.
    Example reducers: sum, append, set_max, custom merge.
    """
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

# Agent base class
class Agent:
    def __init__(self, agent_id: str, bus: MessageBus, blackboard: Blackboard,
                 policy: Policy, llm: LLM, role: str):
        self.id = agent_id
        self.bus = bus
        self.blackboard = blackboard
        self.policy = policy
        self.llm = llm
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.bus.register_agent_inbox(self.id, self.inbox)
        self.topics: Set[str] = set()
        self.running = True
        # subscribe mặc định các kênh cần
        # self.subscribe(MOD_CMD_TOPIC, DRAFT_TOPIC, CRITIC_TOPIC, SEARCH_RES, FINDINGS)
        # cache cục bộ theo session
        self._local: Dict[str, Dict[str, Any]] = {}  # sid -> {invited_for, need_more_sources}

    def subscribe(self, *topics: str):
        for t in topics:
            self.bus.subscribe(t, self.inbox)
            self.topics.add(t)

    async def send(self, topic: str, content: Any, receiver: Optional[str] = None, **meta):
        await self.bus.publish(Message(sender=self.id, topic=topic, content=content, receiver=receiver, meta=meta))

    # ---------- quan sát ----------
    def _local_s(self, sid: str) -> Dict[str, Any]:
        return self._local.setdefault(sid, {"invited_for": None, "need_more_sources": True})

    def build_observation(self, sid: str, st: DebateState) -> Dict[str, Any]:
        loc = self._local_s(sid)
        return {
            "sid": sid,
            "prompt": st.prompt,
            "outline": st.outline,
            "draft": st.draft,
            "critiques": st.critiques,
            "round": st.round,
            "max_rounds": st.max_rounds,
            "invited_for": loc["invited_for"],
            "need_more_sources": loc["need_more_sources"],
            "kb_size": self.blackboard.get(f"kb:{self.id}:{sid}:size", 0),
            "role": self.role,
        }


# Simple Debate Protocol
DEBATE_TOPIC = "debate"
DRAFT_TOPIC = "draft"
CRITIC_TOPIC = "critique"
MOD_CMD_TOPIC = "moderator_cmd"

@dataclass
class DebateState:
    """Lightweight per-session debate memory kept on the blackboard."""
    session_id: str
    prompt: str
    outline: List[str] = field(default_factory=list)
    critiques: List[str] = field(default_factory=list)
    draft: str = ""
    round: int = 0
    max_rounds: int = 2
