#!/bin/bash

# A100의 성능을 활용하기 위해 환경 개수(num-envs)를 16으로 증가
# 학습 총량(total-timesteps)을 100만(1e6)으로 증가

# 1. Shared Reward (팀 보상)
echo "=== Experiment 1: Shared Reward Start (200K steps) ==="
python ppo_overcooked.py \
    --exp-name "Exp1_Shared_Long" \
    --reward-mode "shared" \
    --total-timesteps 200000 \
    --num-envs 32 \
    --seed 1

# 2. Individual Reward (개인 점수)
echo "=== Experiment 2: Individual Reward Start (200K steps) ==="
python ppo_overcooked.py \
    --exp-name "Exp2_Individual_Long" \
    --reward-mode "individual" \
    --total-timesteps 200000 \
    --num-envs 32 \
    --seed 1

# 3. Hybrid Reward (혼합)
echo "=== Experiment 3: Hybrid Reward Start (200K steps) ==="
python ppo_overcooked.py \
    --exp-name "Exp3_Hybrid_Long" \
    --reward-mode "hybrid" \
    --hybrid-alpha 0.5 \
    --total-timesteps 200000 \
    --num-envs 32 \
    --seed 1

echo "All experiments finished!"