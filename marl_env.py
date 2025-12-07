import numpy as np
import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

class MARLEnv(gym.Env):
    """
    Overcooked 환경을 Multi-Agent RL(MARL) 학습에 맞게 감싸는 Wrapper 클래스입니다.
    
    주요 기능:
    1. QMIX 알고리즘에 맞는 State/Action/Reward 구조 제공.
    2. Overcooked-AI 환경의 복잡한 관측값(Observation)을 Tensor로 변환.
    3. 하이브리드 보상(Hybrid Reward) 및 쉐이핑(Shaping) 로직 적용.
    """

    def __init__(self, layout_name="cramped_room", horizon=400, hybrid_alpha=0.5, stay_penalty=0.0, state_mode="simple"):
        """
        초기화 함수.

        Args:
            layout_name (str): 맵 이름 (예: 'cramped_room', 'asymmetric_advantages')
            horizon (int): 한 에피소드의 최대 턴 수 (최적화: 200 추천)
            hybrid_alpha (float): 팀 보상과 개인 보상의 비율 (1.0이면 팀 보상만, 0.0이면 개인 보상만 사용)
            stay_penalty (float): 제자리에 멈춰있을 때 부여할 벌점 (학습 독려용)
            state_mode (str): 'standard'(전체 정보) 또는 'simple'(관측값 연결). 'simple'이 학습 속도가 빠름.
        """
        super().__init__()
        
        # 1. Overcooked 환경 생성
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.env_actions = Action.INDEX_TO_ACTION
        
        # 2. 파라미터 저장
        self.hybrid_alpha = hybrid_alpha
        self.stay_penalty = stay_penalty
        self.state_mode = state_mode
        self.n_agents = 2
        self.episode_limit = horizon
        
        # 3. 보상 쉐이핑 설정 (Reward Shaping Strategy)
        # 에이전트가 특정 행동을 할 때 주는 추가 점수입니다.
        # 이 값들은 OvercookedMDP 내부 로직에 의해 발생 시점에 더해집니다.
        self.shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,      # 냄비에 재료 넣기
            "DISH_PICKUP_REWARD": 3,        # 접시 줍기
            "SOUP_PICKUP_REWARD": 5,        # 완성된 스프 들기
            "DISH_DISP_DISTANCE_REW": 0,    # 거리 기반 보상 (사용 안 함)
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0
        }
        
        # 환경에 쉐이핑 파라미터 주입
        self.mdp.reward_shaping_params = self.shaping_params
        
        # 4. 차원(Dimension) 계산을 위한 더미 실행
        self.env.reset()
        dummy_state = self.env.state
        self.obs_shape = self.get_obs()[0].shape[0]
        
        if self.state_mode == "standard":
            self.state_shape = self.env.lossless_state_encoding_mdp(dummy_state).shape[0]
        else:
            # Simple 모드: 두 에이전트의 관측값을 합쳐서 전역 상태(State)로 사용
            self.state_shape = self.obs_shape * self.n_agents
            
        self.n_actions = len(Action.ALL_ACTIONS)
        
        # 로깅 변수 초기화
        self.cum_sparse_reward = 0  # 실제 게임 점수 (서빙 횟수 * 20)
        self.steps = 0

    def reset(self):
        """에피소드 초기화"""
        self.env.reset()
        
        # MDP가 재생성될 수 있으므로 쉐이핑 파라미터 재주입
        self.env.mdp.reward_shaping_params = self.shaping_params
        
        self.steps = 0
        self.cum_sparse_reward = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """
        환경에서 한 스탭 진행.

        Args:
            actions (list): 각 에이전트의 행동 인덱스 리스트 [a1, a2]

        Returns:
            reward (float): 계산된 보상 (스케일링 적용됨)
            done (bool): 종료 여부
            info (dict): 추가 정보
        """
        # 인덱스 -> Overcooked Action 변환
        joint_action = [self.env_actions[a] for a in actions]
        
        # 환경 실행
        next_state, sparse_reward, done, info = self.env.step(joint_action)
        
        self.steps += 1
        self.cum_sparse_reward += sparse_reward
        
        # --- 보상 계산 로직 (Reward Calculation) ---
        
        # 1. 보상 스케일링 (Reward Scaling)
        # 학습 안정성을 위해 모든 점수를 0.1배로 낮춤 (20점 -> 2.0점)
        reward_scale = 0.1
        
        # 팀 보상 (Team Reward): 서빙 점수 (희소 보상)
        r_team = sparse_reward * reward_scale
        
        # 개인 보상 (Individual Reward): 행동별 쉐이핑 점수
        # info['shaped_r_by_agent']에 환경이 계산한 값이 들어있음
        shaped_r_by_agent = info.get('shaped_r_by_agent', [0.0]*self.n_agents)
        if shaped_r_by_agent is None: shaped_r_by_agent = [0.0]*self.n_agents

        final_rewards = []
        for i in range(self.n_agents):
            r_ind = shaped_r_by_agent[i] * reward_scale
            
            # Hybrid Reward Formula (핵심 로직)
            # R = alpha * (공동 점수) + (1-alpha) * (개인 점수)
            r = (self.hybrid_alpha * r_team) + ((1 - self.hybrid_alpha) * r_ind)
            
            # 패널티 적용: 가만히 있으면 감점 (Stay Penalty)
            if actions[i] == 4: # 4번 액션 = Stay
                r -= self.stay_penalty
                
            # 추가 보정: 양파 줍기 (Onion Pickup)
            # 기존 환경에는 양파 줍기 보상이 없어서 수동으로 추가함
            if 'pickup_onion' in info and info['pickup_onion'][i]:
                 r += 3.0 * reward_scale # 다른 행동과 비슷하게 3점 부여
            
            final_rewards.append(r)
            
        # Terminated vs Truncated
        terminated = done
        if self.steps >= self.episode_limit:
            terminated = True # 이 학습 루프에서는 타임아웃을 종료로 간주

        return final_rewards, terminated, {"team_score": self.cum_sparse_reward}

    def get_obs(self):
        """각 에이전트의 관측값(Observation) 반환"""
        # Overcooked-AI의 featurize_state는 이미 numpy 배열을 반환
        obs = self.mdp.featurize_state(self.env.state, self.env.mlam)
        return [np.array(o, dtype=np.float32) for o in obs]

    def get_state(self):
        # Return global state np array
        if self.state_mode == "standard" and hasattr(self.env, 'lossless_state_encoding_mdp'):
             state = self.env.lossless_state_encoding_mdp(self.env.state)[0]
             return np.array(state, dtype=np.float32).flatten()
        else:
             # Fallback
             obs = self.get_obs()
             return np.concatenate([o.flatten() for o in obs])

    def get_avail_actions(self):
        # Overcooked: All actions always available?
        # Actually in gridworld, walls block movement, but action is still 'valid' (no-op).
        avail = [[1]*self.n_actions for _ in range(self.n_agents)]
        return avail
