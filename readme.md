# QMIX for Overcooked-AI: Cooperative Multi-Agent Reinforcement Learning

본 프로젝트는 **Overcooked-AI** 환경에서 다중 에이전트 강화학습(MARL) 알고리즘인 **QMIX**를 적용하여, 두 에이전트가 협동하여 요리(Soup)를 완성하고 서빙하는 과정을 학습하는 연구 코드입니다.

특히 **Hybrid Reward Structure (하이브리드 보상 체계)**를 도입하여, 개인의 이기적 목표 달성과 팀의 공통 목표 달성 간의 균형(Alpha값)이 학습 효율에 미치는 영향을 분석합니다.

---

## 🏗️ 1. 시스템 구조 및 알고리즘 (System Architecture)

### 1.1 MDP (Markov Decision Process) 설계
*   **환경 (Environment)**: `Overcooked-AI` (layout: `cramped_room`)
*   **상태 (State)**:
    *   **Global State**: 믹싱 네트워크(Mixer)에 입력되는 전역 정보. 두 에이전트의 관측값(Observation)을 연결(Concatenate)하여 사용 (`mode='simple'`).
    *   **Observation**: 각 에이전트의 시야(5x5 등) 및 주방 상태 정보.
*   **행동 (Action Space)**: 6가지 이산 행동 (Discrete)
    *   `0`: 상 (North)
    *   `1`: 하 (South)
    *   `2`: 우 (East)
    *   `3`: 좌 (West)
    *   `4`: 정지 (Stay) - **패널티 대상**
    *   `5`: 상호작용 (Interact) - 줍기/놓기/조리 등
*   **네트워크 (Networks)**:
    *   **Agent (DRQN)**: GRU 기반 순환 신경망. `Input(Obs+LastAction+ID) -> FC -> GRU -> Q-Values`.
    *   **Mixer (HyperNetwork)**: 개별 $Q_a$를 전역 상태 $S$를 기반으로 비선형 결합하여 $Q_{tot}$ 산출. Monotonicity 제약 조건($\frac{\partial Q_{tot}}{\partial Q_a} \ge 0$) 준수.

### 1.2 보상 체계 (Reward Structure) - 핵심 기여
본 프로젝트는 **Hybrid Reward + Shaping + Penalty**의 3중 구조를 갖습니다.

$$ R_{Final} = \color{blue}{[\alpha \cdot R_{Team} + (1-\alpha) \cdot R_{Ind}]} \color{black}{- R_{Stay} + R_{Bonus}} $$

1.  **팀 보상 ($R_{Team}$)**: 서빙 성공 시 **+20점**. (공유)
2.  **개인 보상 ($R_{Ind}$)**: 행동별 쉐이핑 점수. (개별)
    *   냄비에 재료 넣기: **+3점**
    *   접시 줍기: **+3점**
    *   스프 줍기: **+5점**
3.  **알파 ($\alpha, Alpha$)**: 팀/개인 보상 비율 조절 계수.
    *   `0.0`: 완전 개인주의 (이기적)
    *   `0.5`: 중도 (Hybrid)
    *   `1.0`: 완전 전체주의 (팀워크 중심)
4.  **추가 보정**:
    *   **Stay Penalty ($R_{Stay}$)**: 무의미한 정지(Action 4) 시 **-0.01점**.
    *   **Onion Bonus ($R_{Bonus}$)**: 양파 줍기 시 **+3점**. (환경에 기본 부재하여 수동 추가)
5.  **스케일링**: 모든 점수는 신경망 학습 안정성을 위해 **0.1배**로 축소되어 입력됩니다.

---

## 📂 2. 코드 파일 설명 (Codebase)

모든 코드는 학술적 표준에 맞춰 리팩토링되었으며 한글 주석이 포함되어 있습니다.

| 파일명 | 역할 | 주요 내용 |
| :--- | :--- | :--- |
| **`marl_env.py`** | Environment Wrapper | Gym 인터페이스 구현, 보상 계산(Hybrid Formula/Penalty) 로직, 쉐이핑 파라미터 주입. |
| **`qmix_net.py`** | Neural Networks | `RNNAgent`(DRQN) 및 `HyperMixer`(QMIX Mixing Net) 구현. |
| **`qmix_trainer.py`** | Trainer | 에피소드 버퍼(Replay Buffer), 학습 루프(Backprop), 모델 저장. |
| **`run_experiment.py`** | **Orchestrator** | Alpha Sweep (0.0~1.0) 실험 자동화 스크립트. **(실행 파일)** |
| **`plot_metrics.py`** | Analyzer | 실험 결과(`log.csv`)를 로드하여 비교 그래프(`alpha_sweep_results.png`) 생성. |
| **`eval_visualizer.py`** | Visualizer | 학습된 모델(`agent_final.pt`)을 로드하여 GIF 영상 생성. |

---

## 🚀 3. 재연 및 실행 가이드 (Reproduction)

### 3.1 환경 설정 (Requirements)
*   OS: Linux (Recommended)
*   Python: 3.8+
*   Dependencies:
    ```bash
    pip install torch numpy gymnasium matplotlib seaborn imageio pygame
    # Overcooked-AI 설치 필요
    pip install git+https://github.com/HumanCompatibleAI/overcooked_ai.git
    ```

### 3.2 실험 실행 (Run Experiment)
다음 명령어를 실행하면 **5가지 Alpha 값(0, 0.25, 0.5, 0.75, 1)**에 대한 학습이 순차적으로 진행됩니다.

```bash
# 전체 실험 파이프라인 자동 실행 (학습 -> 그래프 -> GIF)
python run_experiment.py
```

*   **소요 시간**: 약 30~60분 (총 10,000~15,000 에피소드)
*   **출력**:
    1.  `results/alpha_0.0/`, `results/alpha_0.5/` 등의 폴더에 로그와 모델 저장.
    2.  `alpha_sweep_results.png`: 학습 곡선 비교 그래프.
    3.  `alpha_0.5.gif`: 에이전트 플레이 영상.

### 3.3 하이퍼파라미터 (Hyperparameters)
빠른 검증을 위해 최적화된 설정입니다 (`run_experiment.py` 내부에서 수정 가능).

*   `n_episodes`: 2000 (충분한 수렴을 보장하는 최소 횟수)
*   `horizon`: 200 step (불필요하게 긴 에피소드 방지)
*   `batch_size`: 128 (안정적 그래디언트)
*   `hidden_dim`: 32 (연산 속도 최적화)
*   `lr`: 0.0005

---

## 📊 4. 결과 분석 (Results Analysis)

실험이 완료되면 루트 디렉토리에 생성된 결과물로 다음을 분석할 수 있습니다.

1.  **팀 점수 (Team Score)**: Alpha 값이 높을수록(팀 보상 비중이 클수록) 협동(서빙)이 잘 이루어지는가?
2.  **개인 행동 빈도**: Alpha 값이 낮을수록 단순한 개인 보상(양파 줍기 등)에만 집착하는가?
3.  **수렴 속도**: Hybrid(0.5)가 극단적인 경우(0.0 또는 1.0)보다 학습이 빠르고 안정적인가?
4.  **시각화 (GIF)**: 실제로 에이전트들이 분업(한 명은 재료 전달, 한 명은 서빙 등)을 수행하는가?
