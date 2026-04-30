#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=2

INNER_LOOP="${INNER_LOOP:-50}"
ITER="${ITER:-300}"
LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-0.15}"
TAGS_CSV="${TAGS_CSV:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --innerLoop)
      INNER_LOOP="$2"
      shift 2
      ;;
    --iter)
      ITER="$2"
      shift 2
      ;;
    --lambda_siglip)
      LAMBDA_SIGLIP="$2"
      shift 2
      ;;
    --tags)
      TAGS_CSV="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--iter N] [--innerLoop N] [--lambda_siglip V] [--tags tag1,tag2,...]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${TAGS_CSV}" ]]; then
  if [[ "${TAGS_CSV}" == *","* ]]; then
    IFS=',' read -r -a TAGS <<< "${TAGS_CSV}"
  else
    read -r -a TAGS <<< "${TAGS_CSV}"
  fi
else
  TAGS=("offline_contact" "phase1R4" "innerLoop${INNER_LOOP}" "gt_siglip")
fi

echo "[run_offline_sanity_pass] ITER=${ITER} INNER_LOOP=${INNER_LOOP} LAMBDA_SIGLIP=${LAMBDA_SIGLIP} TAGS=${TAGS[*]}"

python VLAAttacker/UADA_rollout_wrapper.py \
  --dataset libero_spatial \
  --server /home/ubuntu/yxx/projectAttack \
  --device 4 \
  --bs 8 \
  --iter "${ITER}" \
  --lr 5e-4 \
  --warmup 5 \
  --accumulate 1 \
  --innerLoop "${INNER_LOOP}" \
  --attack_mode projection \
  --patch_size 3,70,70 \
  --projection_size 3,70,70 \
  --projection_alpha 0.8 \
  --projection_alpha_jitter 0.05 \
  --projection_fixed_angle true \
  --projection_shear 0.0 \
  --projection_scale_min 0.9 \
  --projection_scale_max 1.1 \
  --projection_keystone_jitter 0.01 \
  --projection_randomization_enabled true \
  --geometry false \
  --phase1_ratio 1.0 \
  --phase1_rollout 4 \
  --phase2_rollout 8 \
  --action_gap_mode gt_farthest \
  --lambda_action_gap 1.0 \
  --lambda_siglip "${LAMBDA_SIGLIP}" \
  --siglip_model_name google/siglip-so400m-patch14-384 \
  --siglip_device cuda:0 \
  --siglip_input_size 384 \
  --offline_phase_scope contact_manipulate \
  --phase_state_cache_path auto \
  --gt_action_bank_path auto \
  --offline_phase_fallback_enabled true \
  --lambda_history 0.0 \
  --lambda_ce 0.0 \
  --eval_rollout 8 \
  --eval_enabled true \
  --eval_visual_only true \
  --val_max_batches 1 \
  --use_all_joints false \
  --maskidx 0,1,2 \
  --save_interval 10 \
  --sanity_mode false \
  --sanity_num_batches 4 \
  --sanity_disable_randomization false \
  --wandb_entity "1473195970-beihang-university" \
  --wandb_project "projectAttack_sanity_test" \
  --tags "${TAGS[@]}"
