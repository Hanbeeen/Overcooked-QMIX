import torch
import numpy as np
import imageio
from marl_env import MARLEnv
from qmix_net import RNNAgent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import glob
import os
import pygame
from PIL import Image, ImageDraw

def generate_gifs():
    """
    학습된 모델(agent_final.pt)을 로드하여 테스트 에피소드를 실행하고,
    결과를 GIF 파일로 저장합니다.
    """
    results_dir = "results"
    alpha_dirs = glob.glob(f"{results_dir}/alpha_*")
    alpha_dirs.sort()
    
    device = torch.device("cpu") # 시각화는 CPU로 충분
    
    for d in alpha_dirs:
        alpha_val = d.split('_')[-1]
        model_path = os.path.join(d, "agent_final.pt")
        
        if not os.path.exists(model_path):
            print(f"모델 없음: {alpha_val}")
            continue
            
        print(f"GIF 생성 중... Alpha={alpha_val}")
        
        # 1. 검증용 환경 생성 (학습 환경과 동일 설정)
        env = MARLEnv(layout_name="cramped_room", horizon=200, hybrid_alpha=float(alpha_val), stay_penalty=0.01, state_mode="simple")
        
        # 2. 에이전트 모델 설정
        obs_dim = env.obs_shape
        n_actions = env.n_actions
        n_agents = env.n_agents
        input_dim = obs_dim + n_actions + n_agents
        hidden_dim = 32
        
        agent = RNNAgent(input_dim, hidden_dim, n_actions).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        
        # 3. 에피소드 실행 및 렌더링
        frames = []
        hidden_state = agent.init_hidden().expand(n_agents, -1).to(device)
        last_action_onehot = torch.zeros(n_agents, n_actions).to(device)
        
        obs_list, state = env.reset()
        done = False
        
        while not done:
            # 상태 렌더링 (StateVisualizer 사용)
            # Overcooked-AI 내부 시각화 도구 활용
            surface = StateVisualizer().render_state(env.env.state, env.env.mdp.terrain_mtx)
            arr = pygame.surfarray.array3d(surface)
            arr = np.transpose(arr, (1, 0, 2)) # Pygame(W,H,C) -> Image(H,W,C)
            
            # 정보 텍스트 추가 (Pillow 사용)
            pil_img = Image.fromarray(arr)
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), f"Alpha: {alpha_val} | Score: {env.cum_sparse_reward}", fill=(255, 255, 255))
            
            frames.append(np.array(pil_img))
            
            # Action 선택
            obs_tensor = torch.tensor(np.array(obs_list)).float().to(device)
            agent_ids = torch.eye(n_agents).to(device)
            inputs = torch.cat([obs_tensor, last_action_onehot, agent_ids], dim=1)
            
            q_vals, hidden_state = agent(inputs, hidden_state)
            actions = q_vals.argmax(dim=2).detach().numpy() # [N] (Greedy)
            
            # 환경 업데이트
            act_list = [a for a in actions]
            _, done, _ = env.step(act_list)
            
            # 다음 스텝 준비
            actions_oh = np.eye(n_actions)[actions]
            last_action_onehot = torch.tensor(actions_oh).float().to(device)
            
            obs_list = env.get_obs()
            
        # 4. GIF 저장
        out_name = f"alpha_{alpha_val}.gif"
        imageio.mimsave(out_name, frames, fps=15)
        print(f"저장 완료: {out_name}")

if __name__ == "__main__":
    generate_gifs()
