#!/usr/bin/env bash

set -u
shopt -s nullglob

CHECKPOINT_BASE=/weka/oe-training-default/ai2-llm/checkpoints/jacksonp

weight_dirs=(
  "$CHECKPOINT_BASE/hybrid-aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2/60M"
  "$CHECKPOINT_BASE/hybrid-aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/hybrid-periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2/60M"
  "$CHECKPOINT_BASE/hybrid-periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/hybrid-periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/hybrid-r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2/60M"
  "$CHECKPOINT_BASE/hybrid-r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2/60M"
  "$CHECKPOINT_BASE/hybrid-r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2/60M"
  "$CHECKPOINT_BASE/transformer-aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2/60M"
  "$CHECKPOINT_BASE/transformer-aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/transformer-aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/transformer-periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2-init_seed42/60M"
  "$CHECKPOINT_BASE/transformer-periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/transformer-periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4/60M"
  "$CHECKPOINT_BASE/transformer-r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2/60M"
  "$CHECKPOINT_BASE/transformer-r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2/60M"
  "$CHECKPOINT_BASE/transformer-r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2/60M"
)

for weight_dir in "${weight_dirs[@]}"; do
  printf '\n%s\n' "$weight_dir"
  if [[ ! -d "$weight_dir" ]]; then
    echo "MISSING"
  else
    checkpoint_dirs=("$weight_dir"/*/)
    if ((${#checkpoint_dirs[@]} == 0)); then
      echo "EMPTY"
    else
      ls -1d "${checkpoint_dirs[@]}" | sed 's#/$##; s#.*/##' | paste -sd ' ' -
    fi
  fi
done
