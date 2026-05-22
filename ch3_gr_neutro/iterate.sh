#!/bin/bash
# Iteration plan invoked when R1 misses any of the four targets.
# Each "iter_*" submits a single targeted variant.
#
# Usage:
#   bash iterate.sh iter_class_imbalance   # boost rare-class F1
#   bash iterate.sh iter_concept_capacity  # boost concept F1
#   bash iterate.sh iter_balanced_sampler  # alt approach to class imbalance
#   bash iterate.sh iter_more_unfreeze     # bump backbone capacity
#   bash iterate.sh iter_long              # 60 epochs instead of 40
set -euo pipefail
cd $(dirname "$0")

case "${1:-}" in
  iter_class_imbalance)
    TAG=iter_classimb_s2024
    sbatch --job-name=$TAG --export=ALL,TAG=$TAG run_train.sh \
      --tag $TAG --epochs 40 --seed 2024 --lambda_constraint 0.1 \
      --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 \
      --unfreeze_last_n 6 --batch_size 32 --classifier_dropout 0.3 ;;
  iter_concept_capacity)
    TAG=iter_concapacity_s2024
    sbatch --job-name=$TAG --export=ALL,TAG=$TAG run_train.sh \
      --tag $TAG --epochs 40 --seed 2024 --lambda_constraint 0.1 \
      --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 3e-3 \
      --unfreeze_last_n 6 --batch_size 32 --concept_dim 256 --num_heads 8 ;;
  iter_balanced_sampler)
    TAG=iter_balsampler_s2024
    sbatch --job-name=$TAG --export=ALL,TAG=$TAG run_train.sh \
      --tag $TAG --epochs 40 --seed 2024 --lambda_constraint 0.1 \
      --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 \
      --unfreeze_last_n 6 --batch_size 32 --balanced_sampling ;;
  iter_more_unfreeze)
    TAG=iter_unfreeze9_s2024
    sbatch --job-name=$TAG --export=ALL,TAG=$TAG run_train.sh \
      --tag $TAG --epochs 40 --seed 2024 --lambda_constraint 0.1 \
      --lr_backbone 2e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 \
      --unfreeze_last_n 9 --batch_size 32 ;;
  iter_long)
    TAG=iter_long_s2024
    sbatch --job-name=$TAG --export=ALL,TAG=$TAG run_train.sh \
      --tag $TAG --epochs 60 --seed 2024 --lambda_constraint 0.1 \
      --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 \
      --unfreeze_last_n 6 --batch_size 32 ;;
  *)
    echo "Unknown variant: '$1'"
    echo "Options: iter_class_imbalance | iter_concept_capacity | iter_balanced_sampler | iter_more_unfreeze | iter_long"
    exit 1 ;;
esac

squeue -u mouaddenn
