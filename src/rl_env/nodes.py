
from rl_env.states import AgentSpec, SessionLList,MessageBus,Blackboard, Message,SessionLocal
from typing import Any, Dict, Optional, Set, List
from states import DebateState
import asyncio

class Agent:
    def __init__(self, agent_id: str, bus: MessageBus, blackboard: Blackboard, policy, llm, token_budget: int):
        self.id = agent_id
        self.bus = bus
        self.blackboard = blackboard
        self.policy = policy
        self.llm = llm             

        self.token_budget = token_budget 
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.bus.register_agent_inbox(self.id, self.inbox)
        self.running = True
        self.status = "idle"

        self.cur_action: Optional[str] = None
        self.peers: List[str] = []
        self.reward_accum: float = 0.0
        self._local: Dict[str, SessionLocal] = {}  

    def _local_s(self, sid: str) -> SessionLocal:
        if sid not in self._local:
            self._local[sid] = SessionLocal()
        return self._local[sid]

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
            "invited_for": loc.invited_for,
            "need_more_sources": loc.need_more_sources,
            "kb_size": loc.kb_size,
            "role": self.role,
            
            "alpha": self.spec.alpha,
            "beta": self.spec.beta,
            "epsilon": self.spec.epsilon,
            "confidence_threshold": self.spec.confidence_threshold,
            "token_budget": self.spec.token_budget - loc.tokens_used,
        }

    def check_status(self, sid: str) -> bool:
        loc = self._local_s(sid)
        return loc.tokens_used < self.spec.token_budget

    async def write(self, prompt: str, sid: Optional[str] = None,temperature: Optional[float] = None,max_new_tokens: Optional[int] = None) -> str: 
        temp = self.spec.temperature if temperature is None else temperature
        mnt  = self.spec.gen_rate_hint if max_new_tokens is None else max_new_tokens


        if asyncio.iscoroutinefunction(getattr(self.llm, "generate", None)):
            text = await self.llm.generate(prompt, temperature=temp, max_tokens=mnt)
        else:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, lambda: self.llm.generate(prompt, temperature=temp, max_tokens=mnt))

        used = int(len(text.split()) * 1.3) 
            loc = self._local_s(sid)
            loc.tokens_used += used
            loc.history.append({"type": "write", "len": used, "preview": text[:120]})
        return text


