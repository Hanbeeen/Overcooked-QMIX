import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

def plot_all():
    """
    모든 Alpha 실험 결과(log.csv)를 읽어서 비교 그래프를 그립니다.
    """
    results_dir = "results"
    alpha_dirs = glob.glob(f"{results_dir}/alpha_*")
    alpha_dirs.sort()
    
    plt.figure(figsize=(12, 6))
    
    # 1. 팀 점수 그래프 (Team Score)
    plt.subplot(1, 2, 1)
    
    for d in alpha_dirs:
        alpha_val = d.split('_')[-1]
        log_path = os.path.join(d, "log.csv")
        if not os.path.exists(log_path):
            continue
            
        df = pd.read_csv(log_path)
        # 이동 평균(Rolling Mean)으로 그래프를 부드럽게 표현 (Window=20)
        df['score_smooth'] = df['team_score'].rolling(window=20, min_periods=1).mean()
        plt.plot(df['episode'], df['score_smooth'], label=f"Alpha={alpha_val}")
        
    plt.title("Team Score (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Score (Serving)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 총 보상 그래프 (Total Reward)
    plt.subplot(1, 2, 2)
    
    for d in alpha_dirs:
        alpha_val = d.split('_')[-1]
        log_path = os.path.join(d, "log.csv")
        if not os.path.exists(log_path):
            continue
            
        df = pd.read_csv(log_path)
        df['reward_smooth'] = df['train_reward_sum'].rolling(window=20, min_periods=1).mean()
        plt.plot(df['episode'], df['reward_smooth'], label=f"Alpha={alpha_val}")

    plt.title("Total Reward (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward Sum")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("alpha_sweep_results.png")
    print("그래프 저장 완료: alpha_sweep_results.png")

if __name__ == "__main__":
    plot_all()
