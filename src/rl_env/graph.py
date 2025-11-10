import pettingzoo
from pettingzoo import ParallelEnv
import gymnasium.spaces as spaces
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import Dict, Any
from src.rl_env.states import (Blackboard, DebateFeedback, get_observation_space, get_action_space)
from src.agent.action import AgentAction, NUM_ACTIONS 
from agent.nodes import Agent 

from controller.actor import Actor
from controller.critic import Critic
from src.agent.model import WorkerLLM
from src import config
class GraphEnviroment(ParallelEnv):

    metadata = {"name": "rwg_debate_env_v1"}
    def __init__(self):
        super().__init__()
        # Để cập nhật thuộc tính
        self.embedder = SentenceTransformer(config.EMBED_MODEL_NAME)
        self.llm = WorkerLLM()
        self.actors = {aid: Actor() for aid in self.possible_agents}
        self.critics = {aid: Critic() for aid in self.possible_agents}

        # Thuộc tính 
        self.possible_agents = [f"agent_{i}" for i in range(config.NUM_AGENTS)]
        self.agents = []
        self.agent_nodes = {
            aid: Agent(aid, self.actors[aid], self.critics[aid], self.llm)
            for aid in self.possible_agents
        }
        self.state: Blackboard = None
        self.feedback_cache = {}
        self.current_actions = Dict[str, int] = {}

    def observation_space(self, agent: str) -> spaces.Space:
        return get_observation_space()
    
    def action_space(self, agent: str) -> spaces.Space:
        return get_action_space()

    # Reset enviroment
    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        if options is None or "prompt" not in options:
            raise ValueError("Resources 'prompt' in options when resetting the environment.")
            
        self.agents = self.possible_agents[:]
        self.state = Blackboard(
            session_id=str(np.random.randint(1, 10000)),
            prompt=options["prompt"]
        )
        self.feedback_cache = {aid: {"score": 0.0, "critique_text": "Khởi động"} 
                               for aid in self.agents}

        observations = self._get_obs()
        infos = {aid: {} for aid in self.agents}
        return observations, infos

    # Step 
    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        try:
            rewards = self._step_logic(actions)
        except Exception as e:
            rewards = {aid: -10.0 for aid in self.agents}
            
        self.state.round += 1
        
        # Check lại code kiểm tra điều kiện dừng đang để max round
        terminations = {aid: False for aid in self.agents}
        if self.state.round >= self.state.max_rounds:
            terminations = {aid: True for aid in self.agents}
            self.agents = []

        # Điều kiện ngắt quãng đang để là không 
        truncations = {aid: False for aid in self.agents}
        observations = self._get_obs()
        infos = {aid: {} for aid in self.agents}

        return observations, rewards, terminations, truncations, infos

    # Hàm để xây dựng phần thưởng 
    def generate_rewards(self, actions: Dict[str, int]):
        rewards = {aid: 0.0 for aid in self.agents}
        
        for agent_id, action_code in actions.items():
            action = AgentAction(action_code)
            # Phần thương cho wait
            if action != AgentAction.WAIT:
                rewards[agent_id] = -0.05 
            # Phần thưởng cho hành động lên plan
            if action == AgentAction.PLAN:
                rewards[agent_id] = +0.05 
            # Phần thưởng cho hành động viết (Thay bằng rouge score) 
            if action == AgentAction.WRITE:
                # viết lại hàm rough_score
                rough_score = np.random.uniform(0, 0.2)
                rewards[agent_id] += rough_score 
            # Phần thưởng cho hành động search
            if action == AgentAction.SEARCH:
                rewards[agent_id] = +0.03
            # Phần thưởng cho hành động debate
            if action == AgentAction.REVIEW:
                feedback = self._run_debate(self.state.draft)
                self.state.feedback_history.append(feedback)
                self.feedback_cache[agent_id] = {
                    "score": feedback.score,
                    "critique_text": " ".join(feedback.critiques)
                }
                rewards[agent_id] += feedback.score / 10.0       
        # Return một dict các reward tại 1 time step của mỗi agent 
        return rewards
    
    # Xây dựng quan sát (vector) cho tất cả agents
    def get_obs(self) -> Dict[str, dict]:
        """Tạo Quan sát (vector) cho tất cả agents."""
        texts_to_embed = [
            self.state.prompt,
            " ".join(self.state.outline) or "Dàn ý trống",
            self.state.draft or "Chưa có bản nháp",
            " ".join(self.state.knowledge_db) or "Kiến thức trống"
        ]
        
        vectors = self.embedder.encode(texts_to_embed)
        vec_map = {
            "input_vec": vectors[0],
            "outline_vec": vectors[1],
            "document_vec": vectors[2],
            "knowledge_vec": vectors[3],
        }

        all_obs = {}
        for agent_id in self.agents:
            feedback = self.feedback_cache[agent_id]
            score_vec = np.array([feedback["score"]], dtype=np.float32)
            critique_vec = self.embedder.encode([feedback["critique_text"]])[0]
            
            #(Logic để lấy hành động cuối của agent)
            agent_state_vec = np.zeros(NUM_ACTIONS, dtype=np.float32) 
            
            all_obs[agent_id] = {
                **vec_map,
                "agent_state_vec": agent_state_vec,
                "last_feedback_score": score_vec,
                "last_feedback_critique_vec": critique_vec
            }
        return all_obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass