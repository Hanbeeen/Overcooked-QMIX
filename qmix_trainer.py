import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import random
import time
import os
import csv
from marl_env import MARLEnv
from qmix_net import RNNAgent, HyperMixer

class EpisodeBuffer:
    """
    Episode-Replay Buffer (순환 버퍼)
    
    QMIX와 같은 DRQN 기반 알고리즘은 단일 스텝이 아닌 '전체 에피소드'를 저장해야 합니다.
    RNN의 Hidden State를 학습시키기 위해 시간 순서(Sequence)가 보존되어야 하기 때문입니다.
    """
    def __init__(self, capacity, max_episode_len, obs_dim, state_dim, n_agents, n_actions):
        self.capacity = capacity
        self.max_len = max_episode_len
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        # 데이터 저장소 (미리 할당하여 메모리 파편화 방지)
        self.obs = np.zeros((capacity, max_episode_len, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, max_episode_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, max_episode_len, n_agents), dtype=np.int64)
        self.actions_onehot = np.zeros((capacity, max_episode_len, n_agents, n_actions), dtype=np.float32)
        self.rewards = np.zeros((capacity, max_episode_len, n_agents), dtype=np.float32)
        self.terminated = np.zeros((capacity, max_episode_len), dtype=np.float32)
        self.padded = np.ones((capacity, max_episode_len), dtype=np.float32) # 마스킹용 (1=데이터, 0=패딩)
        
        self.count = 0
        self.ptr = 0
        
    def add(self, episode):
        """에피소드 하나를 버퍼에 추가"""
        T = episode['obs'].shape[0]
        idx = self.ptr
        
        self.obs[idx, :T] = episode['obs']
        self.states[idx, :T] = episode['states']
        self.actions[idx, :T] = episode['actions']
        self.actions_onehot[idx, :T] = episode['actions_onehot']
        self.rewards[idx, :T] = episode['rewards']
        self.terminated[idx, :T] = episode['terminated']
        
        # 실제 데이터가 있는 곳은 1, 나머지는 0 (RNN 학습 시 무시하기 위함)
        self.padded[idx, :T] = 1.0
        self.padded[idx, T:] = 0.0
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size):
        """무작위 배치를 샘플링 (Tensor 변환 포함)"""
        idxs = np.random.choice(self.count, batch_size, replace=False)
        return {
            'obs': torch.tensor(self.obs[idxs]).float(),
            'states': torch.tensor(self.states[idxs]).float(),
            'actions': torch.tensor(self.actions[idxs]).long(),
            'actions_onehot': torch.tensor(self.actions_onehot[idxs]).float(),
            'rewards': torch.tensor(self.rewards[idxs]).float(),
            'terminated': torch.tensor(self.terminated[idxs]).float(),
            'padded': torch.tensor(self.padded[idxs]).float() # Loss 마스킹용
        }

def train(args):
    """메인 학습 루프"""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 1. 환경 초기화 (Environment Setup)
    env = MARLEnv(
        layout_name=args.layout, 
        horizon=args.horizon, 
        hybrid_alpha=args.hybrid_alpha, 
        stay_penalty=args.stay_penalty, 
        state_mode=args.state_mode
    )
    obs_dim = env.obs_shape
    state_dim = env.state_shape
    n_agents = env.n_agents
    n_actions = env.n_actions
    
    # 2. 네트워크 초기화 (DRQN + Mixer)
    # 에이전트 입력 = [관측값 + 이전 행동 + 에이전트 ID] (식별력 강화)
    agent_input_dim = obs_dim + n_actions + n_agents
    
    agent = RNNAgent(agent_input_dim, args.hidden_dim, n_actions).to(device)
    target_agent = RNNAgent(agent_input_dim, args.hidden_dim, n_actions).to(device)
    target_agent.load_state_dict(agent.state_dict())
    
    mixer = HyperMixer(n_agents, state_dim).to(device)
    target_mixer = HyperMixer(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())
    
    # Optimizer (전체 파라미터 등록)
    params = list(agent.parameters()) + list(mixer.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    
    # Replay Buffer
    buffer = EpisodeBuffer(args.buffer_size, args.horizon, obs_dim, state_dim, n_agents, n_actions)
    
    # 3. 로깅 디렉토리 생성
    # 예: results/alpha_0.5 (Alpha Sweep 실험용 이름)
    run_name = f"alpha_{args.hybrid_alpha}"
    log_dir = f"results/{run_name}"
    if os.path.exists(log_dir):
        print(f"Warning: {log_dir} already exists.")
    os.makedirs(log_dir, exist_ok=True)
    
    # CSV 로거
    csv_f = open(f"{log_dir}/log.csv", "w")
    writer = csv.writer(csv_f)
    writer.writerow(['episode', 'epsilon', 'train_reward_sum', 'team_score'])
    
    epsilon = args.epsilon_start
    
    print(f"Start Training: {run_name}")
    print(f"Alpha: {args.hybrid_alpha}, StayPenalty: {args.stay_penalty}")
    
    # --- Training Loop ---
    for episode in range(args.n_episodes):
        # 환경 리셋
        obs_list, state = env.reset()
        terminated = False
        
        # 에피소드 데이터 저장을 위한 임시 리스트
        ep_obs, ep_state, ep_action, ep_action_onehot, ep_reward, ep_term = [], [], [], [], [], []
        
        # RNN Hidden State 초기화
        hidden_state = agent.init_hidden().expand(n_agents, -1).to(device)
        last_action_onehot = torch.zeros(n_agents, n_actions).to(device)
        
        ep_r_sum = 0
        
        while not terminated:
            # 1. Action 선택 (Epsilon-Greedy)
            obs_tensor = torch.tensor(np.array(obs_list)).float().to(device)
            agent_ids = torch.eye(n_agents).to(device) # 원-핫 ID
            
            # 입력 결합: [Obs, LastAct, AgentID]
            inputs = torch.cat([obs_tensor, last_action_onehot, agent_ids], dim=1)
            
            # Action Q값 계산
            q_vals, hidden_state = agent(inputs, hidden_state)
            
            actions = []
            for i in range(n_agents):
                if np.random.random() < epsilon:
                    actions.append(np.random.randint(n_actions)) # 탐험 (Exploration)
                else:
                    actions.append(q_vals[i].argmax().item())    # 이용 (Exploitation)
            
            # 2. 환경 진행 (Step)
            rewards, terminated, info = env.step(actions)
            next_obs_list, next_state = env.get_obs(), env.get_state()
            
            # Action One-Hot 변환 (다음 스텝 입력용)
            actions_oh = np.eye(n_actions)[actions]
            last_action_onehot = torch.tensor(actions_oh).float().to(device)
            
            # 데이터 저장
            ep_obs.append(obs_list)
            ep_state.append(state)
            ep_action.append(actions)
            ep_action_onehot.append(actions_oh)
            ep_reward.append(rewards)
            ep_term.append(1 if terminated else 0)
            
            ep_r_sum += sum(rewards)
            
            obs_list = next_obs_list
            state = next_state
        
        # 버퍼에 에피소드 저장
        buffer.add({
            'obs': np.array(ep_obs),
            'states': np.array(ep_state),
            'actions': np.array(ep_action),
            'actions_onehot': np.array(ep_action_onehot),
            'rewards': np.array(ep_reward),
            'terminated': np.array(ep_term)
        })
        
        # Epsilon 감쇠 (Linear Decay)
        if epsilon > args.epsilon_end:
            epsilon -= (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_steps
            
        # Logging
        if episode % 10 == 0:
            print(f"Ep {episode} | Score: {info['team_score']} | Train R: {ep_r_sum:.2f} | Eps: {epsilon:.3f}")
            writer.writerow([episode, epsilon, ep_r_sum, info['team_score']])
            csv_f.flush()
            
        # --- Neural Network Update (Train) ---
        if buffer.count >= args.batch_size and episode % 1 == 0:
            batch = buffer.sample(args.batch_size)
            
            # GPU/CPU 이동
            b_obs = batch['obs'].to(device)
            b_states = batch['states'].to(device)
            b_actions = batch['actions'].to(device)
            b_actions_oh = batch['actions_onehot'].to(device)
            b_rewards = batch['rewards'].to(device)
            b_terminated = batch['terminated'].to(device)
            b_mask = batch['padded'].to(device)
            
            bs, max_t, _ = b_rewards.shape
            
            # 1. 입력 시퀀스 구성
            # Last Action: t=0일 땐 0, t>0일 땐 이전 행동
            last_actions = torch.zeros(bs, max_t, n_agents, n_actions).to(device)
            last_actions[:, 1:, :, :] = b_actions_oh[:, :-1, :, :]
            
            agent_ids = torch.eye(n_agents).to(device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
            inputs = torch.cat([b_obs, last_actions, agent_ids], dim=3)
            
            # 2. Agent Network (Online & Target) Forward
            # 시계열 데이터를 처리하기 위해 루프 사용 (Hidden State 전파)
            hidden = torch.zeros(bs * n_agents, args.hidden_dim).to(device)
            target_hidden = torch.zeros(bs * n_agents, args.hidden_dim).to(device)
            
            q_evals = [] # Main Q Network
            q_targets = [] # Target Q Network
            
            for t in range(max_t):
                inp = inputs[:, t].reshape(-1, agent_input_dim)
                
                # Main Net
                q, hidden = agent(inp, hidden)
                q_evals.append(q.reshape(bs, n_agents, n_actions))
                
                # Target Net
                with torch.no_grad():
                    q_tar, target_hidden = target_agent(inp, target_hidden)
                    q_targets.append(q_tar.reshape(bs, n_agents, n_actions))
            
            q_evals = torch.stack(q_evals, dim=1)     # [Batch, Time, N, Actions]
            q_targets = torch.stack(q_targets, dim=1) # [Batch, Time, N, Actions]
            
            # 3. Q값 선택
            # 에이전트가 실제로 수행한 행동의 Q값 추출
            chosen_action_q = torch.gather(q_evals, 3, b_actions.unsqueeze(3)).squeeze(3)
            
            # Double DQN Logic: Action Selection은 Main Net, Valuation은 Target Net
            # Target Q값 계산 (t+1 시점)
            target_max_q_val = q_targets.max(dim=3)[0] # Simplified Max (Standard VDN/QMIX style)
            # 엄밀한 Double DQN: target_max_q_val = q_targets.gather(3, q_evals.argmax(3).unsqueeze(3)).squeeze(3)
            # 여기선 간단히 Max 사용 (일반적 구현)
            
            # 4. Mixer Forward
            # 개별 $Q_a$를 통합 $Q_{tot}$로 변환
            
            # 현재 시점 (t) 평가용
            q_tot_eval = mixer(chosen_action_q[:, :-1].reshape(-1, 1, n_agents), 
                               b_states[:, :-1].reshape(-1, state_dim))
            q_tot_eval = q_tot_eval.reshape(bs, -1, 1)
            
            # 다음 시점 (t+1) 타겟용
            with torch.no_grad():
                q_tot_target = target_mixer(target_max_q_val[:, 1:].reshape(-1, 1, n_agents), 
                                            b_states[:, 1:].reshape(-1, state_dim))
                q_tot_target = q_tot_target.reshape(bs, -1, 1)
                
            # 5. Loss Calculation
            # Q_tot_target = Reward + Gamma * Q_tot_next
            # 주의: QMIX는 "팀 전체 보상의 합"을 예측하도록 학습됨
            rewards_sum = b_rewards[:, :-1].sum(dim=2, keepdim=True)
            mask = b_mask[:, :-1].unsqueeze(2)
            term_mask = b_terminated[:, :-1].unsqueeze(2)
            
            targets = rewards_sum + args.gamma * (1 - term_mask) * q_tot_target
            td_error = (q_tot_eval - targets)
            
            # 마스킹된 평균 제곱 오차 (MSE Loss)
            loss = ((td_error ** 2) * mask).sum() / mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_norm_clip)
            optimizer.step()
            
            # Target Network Soft/Hard Update
            if episode % args.target_update_interval == 0:
                target_agent.load_state_dict(agent.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                
        # 모델 저장
        if episode > 0 and episode % 500 == 0:
            torch.save(agent.state_dict(), f"{log_dir}/agent_{episode}.pt")
            
    torch.save(agent.state_dict(), f"{log_dir}/agent_final.pt")
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overcooked QMIX Trainer")
    
    # 환경 설정
    parser.add_argument("--layout", default="cramped_room", help="Overcooked 맵 이름")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Alpha값 (0.0=개인중심 ~ 1.0=팀중심)")
    parser.add_argument("--stay-penalty", type=float, default=0.005, help="정지 패널티")
    parser.add_argument("--state-mode", default="simple", choices=["standard", "simple"], help="상태 표현 방식")
    
    # 학습 하이퍼파라미터 (Fast Training 설정)
    parser.add_argument("--n-episodes", type=int, default=2000, help="총 학습 에피소드 수")
    parser.add_argument("--horizon", type=int, default=200, help="에피소드 길이")
    parser.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    parser.add_argument("--buffer-size", type=int, default=5000, help="리플레이 버퍼 크기")
    parser.add_argument("--lr", type=float, default=5e-4, help="학습률")
    parser.add_argument("--hidden-dim", type=int, default=32, help="네트워크 은닉층 크기")
    parser.add_argument("--gamma", type=float, default=0.99, help="할인율")
    
    # 기타
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1500)
    parser.add_argument("--grad-norm-clip", type=float, default=10.0)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--cuda", action="store_true", default=True, help="GPU 사용 여부")
    
    args = parser.parse_args()
    train(args)
