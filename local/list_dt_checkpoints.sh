#!/usr/bin/env bash

set -u
shopt -s nullglob

CHECKPOINT_BASE=/weka/oe-training-default/ai2-llm/model-ladders/synthetic-ladder/60M

model_types=(hybrid transformer)
dataset_dirs=(
  r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5
  aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10
  periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10
)
seeds=(0 7 42)

for model_type in "${model_types[@]}"; do
  for dataset_dir in "${dataset_dirs[@]}"; do
    for seed in "${seeds[@]}"; do
      weight_dir="$CHECKPOINT_BASE/$model_type/$dataset_dir/Cx2/init_seed$seed"
      printf '\n%s\n' "$weight_dir"

      if [[ ! -d "$weight_dir" ]]; then
        echo "MISSING"
        continue
      fi

      checkpoint_dirs=("$weight_dir"/step*-hf/)
      if ((${#checkpoint_dirs[@]} == 0)); then
        echo "EMPTY"
      else
        ls -1d "${checkpoint_dirs[@]}" | sed 's#/$##; s#.*/##' | paste -sd ' ' -
      fi
    done
  done
done
