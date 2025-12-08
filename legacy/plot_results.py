import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ---------------------------------------------------------
# 1. 설정 및 파일 탐색
# ---------------------------------------------------------
# runs 폴더 하위의 모든 폴더에서 training_log.csv를 찾습니다.
# 구조: runs/{실험명}/training_log.csv
log_files = glob.glob(os.path.join('runs', '*', 'training_log.csv'))

if not log_files:
    print("Error: 'runs' 폴더 내에서 training_log.csv 파일을 찾을 수 없습니다.")
    exit()

print(f"총 {len(log_files)}개의 로그 파일을 발견했습니다.")

# ---------------------------------------------------------
# 2. 이동 평균 (Smoothing) 함수
# ---------------------------------------------------------
def rolling_average(data, window_size=50):
    """그래프를 부드럽게 만들기 위한 이동 평균"""
    return data.rolling(window=window_size, min_periods=1).mean()

# ---------------------------------------------------------
# 3. 그래프 그리기 준비
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# 실험 모드별 색상 지정 (그래프 가독성을 위해)
colors = {
    'shared': 'blue',
    'individual': 'red',
    'hybrid': 'green'
}

# ---------------------------------------------------------
# 4. 각 로그 파일 읽어서 플롯
# ---------------------------------------------------------
for file_path in log_files:
    try:
        # 폴더 이름에서 실험 모드(shared, individual, hybrid) 추출
        # 폴더명 예시: cramped_room__hybrid__1__1764550483
        folder_name = os.path.basename(os.path.dirname(file_path))
        parts = folder_name.split('__')
        
        # 파일명 규칙에 따라 모드 추출 (예외 처리 포함)
        if len(parts) >= 2:
            mode = parts[1]  # hybrid, individual, shared
        else:
            mode = folder_name

        label_name = f"{mode.upper()}"
        color = colors.get(mode, 'gray') # 지정되지 않은 모드는 회색

        # CSV 읽기
        df = pd.read_csv(file_path)
        
        # 윈도우 사이즈 자동 조절 (데이터 길이에 따라)
        window = int(len(df) * 0.05) if len(df) > 100 else 5

        # (1) Mean Step Reward
        axes[0].plot(df['global_step'], rolling_average(df['mean_step_reward'], window), 
                     label=label_name, color=color, linewidth=2, alpha=0.8)

        # (2) Episodic Return
        # 0이 아닌 값들만 필터링해서 보거나, 전체 추세를 봅니다.
        axes[1].plot(df['global_step'], rolling_average(df['episodic_return'], window), 
                     label=label_name, color=color, linewidth=2, alpha=0.8)

        # (3) Episodic Length
        axes[2].plot(df['global_step'], rolling_average(df['episodic_length'], window), 
                     label=label_name, color=color, linewidth=2, alpha=0.8)
        
        print(f"--> Loaded: {folder_name}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# ---------------------------------------------------------
# 5. 그래프 꾸미기 및 저장
# ---------------------------------------------------------

# 첫 번째 그래프: Mean Step Reward
axes[0].set_title('Mean Step Reward (Training Progress)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Reward per Step', fontsize=12)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 두 번째 그래프: Episodic Return
axes[1].set_title('Episodic Return (Total Score)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Total Score', fontsize=12)
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# 세 번째 그래프: Episodic Length
axes[2].set_title('Episodic Length', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Steps', fontsize=12)
axes[2].set_xlabel('Global Steps', fontsize=12)
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# 결과 저장
output_file = 'comparison_results.png'
plt.savefig(output_file, dpi=300)
print(f"\n그래프가 '{output_file}'로 저장되었습니다. 파일 탐색기에서 확인하세요!")
# plt.show() # 서버 환경이라 창을 띄울 수 없으면 주석 처리 유지