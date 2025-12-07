import os
# [핵심] 서버에서 화면 없이 Pygame을 돌리기 위한 설정
os.environ["SDL_VIDEODRIVER"] = "dummy"

import glob
import torch
import torch.nn as nn
import numpy as np
import imageio
import gymnasium as gym
import pygame
from torch.distributions.categorical import Categorical
from PIL import Image, ImageDraw, ImageFont

# Overcooked 관련 임포트
from overcooked_wrapper import OvercookedPPOWrapper
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        in_channels = obs_shape[0]
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            temp_net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            out_dim = temp_net(dummy_input).shape[1]

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(out_dim, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, envs.single_action_space.n)
        self.critic = nn.Linear(512, 1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action

def save_gif(exp_name, layout_name="cramped_room"):
    run_dir = f"runs/{exp_name}"
    model_path = f"{run_dir}/agent.pt"
    
    if not os.path.exists(model_path):
        print(f"Skipping (No model found): {run_dir}")
        return

    print(f"Processing: {exp_name}...")

    # 모델 구조 로드를 위한 더미 벡터 환경
    try:
        dummy_envs = gym.vector.SyncVectorEnv([lambda: OvercookedPPOWrapper(layout_name=layout_name)])
    except Exception as e:
        print(f"Env Error: {e}")
        return

    # 에이전트 로드
    device = torch.device("cpu")
    agent = Agent(dummy_envs).to(device)
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    # [중요] 시각화용 단일 환경 생성 (Wrapper 직접 사용)
    env = OvercookedPPOWrapper(layout_name=layout_name)
    visualizer = StateVisualizer()
    
    obs, _ = env.reset()
    frames = []
    
    # 400 스텝 (1 에피소드) 진행
    for step in range(400):
        # 1. 렌더링 및 텍스트 오버레이
        try:
            # Overcooked 화면 그리기
            surface = visualizer.render_state(env.base_env.state, grid=env.base_env.mdp.terrain_mtx)
            data = pygame.surfarray.array3d(surface)
            data = data.transpose([1, 0, 2]) # (H, W, C)
            
            # Pillow로 변환하여 텍스트 추가
            img = Image.fromarray(data)
            draw = ImageDraw.Draw(img)
            
            # [수정됨] 실제 게임 스코어 가져오기 (Wrapper 내부 변수 접근)
            real_score = env.cum_sparse_reward
            
            # 텍스트 내용
            text = f"Step: {step}/400\nReal Score: {real_score}"
            
            # 텍스트 위치 및 색상 (좌측 상단, 흰글씨+검은테두리)
            x, y = 10, 10
            # 그림자 효과 (가독성 향상)
            draw.text((x+1, y+1), text, fill=(0, 0, 0)) 
            draw.text((x, y), text, fill=(255, 255, 255)) 
            
            frames.append(np.array(img))
            
        except Exception as e:
            # 렌더링 실패해도 학습 진행은 계속
            pass

        # 2. 행동 선택
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.get_action_and_value(obs_tensor)
        
        # 3. 환경 진행
        obs, reward, done, truncated, info = env.step(action.item())
        
        if done or truncated:
            break

    # GIF 저장
    # 파일명에 최종 실제 점수 포함
    final_score = env.cum_sparse_reward
    save_path = f"{exp_name}_score{final_score}.gif"
    
    if len(frames) > 0:
        try:
            imageio.mimsave(save_path, frames, fps=15, loop=0)
            print(f"--> Saved GIF: {save_path}")
        except Exception as e:
            print(f"Save Error: {e}")
    else:
        print("--> Failed to record frames.")

if __name__ == "__main__":
    if not os.path.exists("runs"):
        print("'runs' folder not found!")
    else:
        runs = sorted(glob.glob("runs/*"))
        print(f"Found {len(runs)} directories in runs/")
        
        for run_folder in runs:
            exp_name = os.path.basename(run_folder)
            save_gif(exp_name)