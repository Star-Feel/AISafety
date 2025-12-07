#!/bin/bash

# 可用攻击列表
ATTACKS=('bsr' 'ops' 'bfa' 'p2fa' 'ana' 'smer' 'mef' 'mumodig' 'gaa' 'foolmix' 'anda')

# 固定参数
BASE_CMD="cd /media/homework/AISAFETY/AISafety/TransferAttack && python pipeline.py \
  --clean_dir datasets/CIFAR10_clean \
  --cifar_root datasets/CIFAR10 \
  --skip_finetune \
  --attack_ensembles Bartoldson2024Adversarial_WRN-94-16+Debenedetti2022Light_XCiT-L12+Bai2023Improving_edm \
  --robust_models Bartoldson2024Adversarial_WRN-94-16 Debenedetti2022Light_XCiT-L12 Bai2023Improving_edm \
  --robust_dataset cifar10 \
  --robust_threat_model Linf \
  --batchsize 20 \
  --gpu 1"

LOG_DIR="attack_logs"
mkdir -p ${LOG_DIR}

# 遍历所有可能的 attack1 -> attack2 组合
for a in "${ATTACKS[@]}"; do
    for b in "${ATTACKS[@]}"; do
        if [[ "$a" != "$b" ]]; then
            PAIR="${a}->${b}"
            echo "===== Running attack pair: ${PAIR} ====="

            # 运行命令并输出到日志
            bash -c "${BASE_CMD} --attacks \"${PAIR}\"" \
              | tee "${LOG_DIR}/${a}_${b}.log"
        fi
    done
done

echo "全部组合攻击运行完成！日志保存在 ${LOG_DIR}/"
