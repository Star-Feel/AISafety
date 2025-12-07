#!/bin/bash

# 全部攻击列表
ATTACKS=('bsr' 'ops' 'bfa' 'p2fa' 'ana' 'smer' 'mef' 'mumodig' 'gaa' 'foolmix' 'anda')

# 高效组合（重复次数）
HIGH_EFFICIENCY=("mumodig->foolmix")
HIGH_REPEAT=4   # 重复次数，可修改

# 固定参数（gpu 后面动态指定）
CMD_PREFIX="cd /media/homework/AISAFETY/AISafety/TransferAttack && python pipeline.py \
  --clean_dir datasets/CIFAR10_clean \
  --cifar_root datasets/CIFAR10 \
  --skip_finetune \
  --attack_ensembles Bartoldson2024Adversarial_WRN-94-16+Debenedetti2022Light_XCiT-L12+Bai2023Improving_edm \
  --robust_models Bartoldson2024Adversarial_WRN-94-16 Debenedetti2022Light_XCiT-L12 Bai2023Improving_edm \
  --robust_dataset cifar10 \
  --robust_threat_model Linf \
  --batchsize 20"

LOG_DIR="attack_logs_weighted"
mkdir -p ${LOG_DIR}

# ========== 先加入高效组合 ==========
COMBOS=()
for combo in "${HIGH_EFFICIENCY[@]}"; do
    for ((i=0;i<HIGH_REPEAT;i++)); do
        COMBOS+=("$combo")
    done
done

# ========== 随机生成剩余组合 ==========
TARGET_TOTAL=25
while [ ${#COMBOS[@]} -lt $TARGET_TOTAL ]; do
    a=${ATTACKS[$RANDOM % ${#ATTACKS[@]}]}
    b=${ATTACKS[$RANDOM % ${#ATTACKS[@]}]}
    c=${ATTACKS[$RANDOM % ${#ATTACKS[@]}]}

    if [[ "$a" != "$b" && "$b" != "$c" && "$a" != "$c" ]]; then
        combo="${a}->${b}->${c}"
        [[ " ${COMBOS[*]} " =~ " ${combo} " ]] || COMBOS+=("$combo")
    fi
done

echo "==== 共生成 25 个攻击组合（高效组合重复 ${HIGH_REPEAT} 次）===="
printf '%s\n' "${COMBOS[@]}"

# ========== 分配 GPU ==========
NUM_COMBOS=${#COMBOS[@]}
NUM_GPU0=$((NUM_COMBOS/2))
GPU0_LIST=("${COMBOS[@]:0:NUM_GPU0}")
GPU1_LIST=("${COMBOS[@]:NUM_GPU0}")

echo "GPU0 分配: ${GPU0_LIST[@]}"
echo "GPU1 分配: ${GPU1_LIST[@]}"

run_task () {
    combo=$1
    gpu=$2
    log="${LOG_DIR}/${combo//->/_}.log"
    echo "启动任务：${combo} 在 GPU${gpu}"
    bash -c "${CMD_PREFIX} --gpu ${gpu} --attacks \"${combo}\"" &> "${log}" &
}

# ========== 并发运行 GPU0 的任务 ==========
for combo in "${GPU0_LIST[@]}"; do
    run_task "${combo}" 0
done

# ========== 并发运行 GPU1 的任务 ==========
for combo in "${GPU1_LIST[@]}"; do
    run_task "${combo}" 1
done

echo "所有任务已启动并发运行！"
echo "使用: ps -ef | grep pipeline.py 查看进度"
echo "日志在 ${LOG_DIR}/"
