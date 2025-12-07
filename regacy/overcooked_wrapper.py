import numpy as np
import gymnasium as gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

class OvercookedPPOWrapper(gym.Env):
    def __init__(self, layout_name="cramped_room", reward_mode="hybrid", hybrid_alpha=0.5):
        super().__init__()
        
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        # horizon=400: 충분한 시간 제공
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.reward_mode = reward_mode
        self.hybrid_alpha = hybrid_alpha

        self.action_space = gym.spaces.Discrete(36)
        self.env_actions = Action.INDEX_TO_ACTION 
        self.action_mapping = [
            (self.env_actions[a1], self.env_actions[a2]) 
            for a1 in range(6)
            for a2 in range(6)
        ]

        self.encode_fn = self._detect_encoding_fn()
        
        # 실제 점수 추적용 변수
        self.cum_sparse_reward = 0
        
        self.base_env.reset()
        dummy_obs = self._get_obs(self.base_env.state)
        
        self.channels = dummy_obs.shape[0]
        self.width = dummy_obs.shape[1]
        self.height = dummy_obs.shape[2]

        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, 
            shape=(self.channels, self.width, self.height), 
            dtype=np.float32
        )

    def _detect_encoding_fn(self):
        if hasattr(self.mdp, 'get_lossless_encoding_vector'):
            return self.mdp.get_lossless_encoding_vector
        if hasattr(self.base_env, 'lossless_state_encoding_mdp'):
            return self.base_env.lossless_state_encoding_mdp
        raise AttributeError("Overcooked 상태 인코딩 함수를 찾을 수 없습니다.")

    def step(self, action_idx):
        joint_action = self.action_mapping[action_idx]
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)
        
        # [핵심] 실제 게임 점수(Sparse Reward) 누적
        self.cum_sparse_reward += sparse_reward

        info = info.copy()
        if "episode" in info: del info["episode"]

        # 에피소드 종료 시 실제 점수를 info에 기록하고 초기화
        if done:
            info["actual_score"] = self.cum_sparse_reward
            self.cum_sparse_reward = 0 # [중요] 점수 리셋!

        agent_shaped = info.get('shaped_r_by_agent', [0.0, 0.0])
        if agent_shaped is None: agent_shaped = [0.0, 0.0]
        
        final_reward = 0.0
        
        if self.reward_mode == "shared":
            final_reward = sparse_reward
        elif self.reward_mode == "individual":
            final_reward = sum(agent_shaped)
        elif self.reward_mode == "hybrid":
            team_score = sparse_reward
            indiv_score = sum(agent_shaped)
            final_reward = (self.hybrid_alpha * team_score) + ((1 - self.hybrid_alpha) * indiv_score)

        terminated = False 
        truncated = done 

        return self._get_obs(next_state), final_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.base_env.reset()
        # 리셋 시 점수 초기화
        self.cum_sparse_reward = 0
        return self._get_obs(self.base_env.state), {}

    def _get_obs(self, state):
        obs = self.encode_fn(state)
        if isinstance(obs, tuple):
            obs = obs[0]
        return np.moveaxis(obs, -1, 0).astype(np.float32)