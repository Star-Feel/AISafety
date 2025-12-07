#!/bin/bash

cd /media/homework/AISAFETY/AISafety/TransferAttack || exit

CLEAN_DIR="datasets/CIFAR10_clean"
CIFAR_ROOT="datasets/CIFAR10"
ROBUST_MODELS="Bartoldson2024Adversarial_WRN-94-16 Debenedetti2022Light_XCiT-L12 Bai2023Improving_edm"
ENSEMBLE="Bartoldson2024Adversarial_WRN-94-16+Debenedetti2022Light_XCiT-L12+Bai2023Improving_edm"

INPUT_ATTACK="ops"
GRAD_ATTACKS=("mumodig" "gaa" "foolmix")
MODEL_ATTACK="ana"

LOG_DIR="attack_logs_ops"
mkdir -p ${LOG_DIR}

# ==========================
# 两两组合梯度攻击：ops->grad1->grad2
# ==========================
echo "=== Three-stage attacks (ops -> grad1 -> grad2) ==="

for i in "${!GRAD_ATTACKS[@]}"; do
    for j in "${!GRAD_ATTACKS[@]}"; do
        if [ "$i" -ne "$j" ]; then
            atk="${INPUT_ATTACK}->${GRAD_ATTACKS[i]}->${GRAD_ATTACKS[j]}"
            echo "[Running] Attack: $atk"
            python pipeline.py \
              --clean_dir "$CLEAN_DIR" \
              --cifar_root "$CIFAR_ROOT" \
              --skip_finetune \
              --attack_ensembles "$ENSEMBLE" \
              --robust_models $ROBUST_MODELS \
              --robust_dataset cifar10 \
              --robust_threat_model Linf \
              --batchsize 10 \
              --attacks "$atk" \
              --gpu 0 \
              &> "${LOG_DIR}/${atk//->/_}.log"
        fi
    done
done

# ==========================
# 可选四段式：ops->grad1->grad2->ana
# ==========================
echo "=== Four-stage attacks (ops -> grad1 -> grad2 -> model) ==="

for i in "${!GRAD_ATTACKS[@]}"; do
    for j in "${!GRAD_ATTACKS[@]}"; do
        if [ "$i" -ne "$j" ]; then
            atk="${INPUT_ATTACK}->${GRAD_ATTACKS[i]}->${GRAD_ATTACKS[j]}->${MODEL_ATTACK}"
            echo "[Running] Attack: $atk"
            python pipeline.py \
              --clean_dir "$CLEAN_DIR" \
              --cifar_root "$CIFAR_ROOT" \
              --skip_finetune \
              --attack_ensembles "$ENSEMBLE" \
              --robust_models $ROBUST_MODELS \
              --robust_dataset cifar10 \
              --robust_threat_model Linf \
              --batchsize 10 \
              --attacks "$atk" \
              --gpu 0 \
              &> "${LOG_DIR}/${atk//->/_}.log"
        fi
    done
done
