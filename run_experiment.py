import subprocess
import os
import sys

# --- 실험 설정 (Experiment Configuration) ---
# Alpha값 비교: 0 (개인주의) ~ 1 (전체주의)
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

# 학습 파라미터
n_episodes = 2000  # 빠른 비교를 위해 2000 에피소드 설정
stay_penalty = 0.01

python_exec = sys.executable

print("====================================")
print(f"Alpha Sweep 실험 시작: {alphas}")
print("====================================")

# 1. Alpha별 순차 학습 진행
for alpha in alphas:
    print(f"\n>>> [학습 시작] Alpha = {alpha}")
    
    # qmix_trainer.py 실행 명령 구성
    cmd = [
        python_exec, "qmix_trainer.py",
        "--hybrid-alpha", str(alpha),
        "--n-episodes", str(n_episodes),
        "--stay-penalty", str(stay_penalty),
        "--state-mode", "simple", # 학습 속도 최적화
        "--horizon", "200",       # 에피소드 길이 단축
        "--batch-size", "128",    # GPU 효율 증대
        "--hidden-dim", "32"      # 모델 경량화
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f">>> [학습 완료] Alpha {alpha}")
    except subprocess.CalledProcessError as e:
        print(f"!!! [오류 발생] Alpha {alpha}: {e}")

print("\n====================================")
print("모든 학습이 완료되었습니다.")
print("결과 분석 및 시각화를 시작합니다...")
print("====================================")

# 2. 결과 그래프 그리기 (Comparison Plot)
try:
    subprocess.run([python_exec, "plot_metrics.py"], check=True)
    print(">>> [완료] 결과 그래프 저장됨 (alpha_sweep_results.png)")
except Exception as e:
    print(f"!!! [오류] 그래프 그리기 실패: {e}")

# 3. 에이전트 행동 시각화 (GIF Generation)
try:
    subprocess.run([python_exec, "eval_visualizer.py"], check=True)
    print(">>> [완료] GIF 생성 완료")
except Exception as e:
    print(f"!!! [오류] 시각화 실패: {e}")
