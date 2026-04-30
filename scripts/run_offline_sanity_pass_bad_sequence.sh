#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=2

INNER_LOOP="${INNER_LOOP:-50}"
ITER="${ITER:-300}"
LAMBDA_SIGLIP="${LAMBDA_SIGLIP:-0.15}"
BADSEQ_ALPHA="${BADSEQ_ALPHA:-1.5}"
LAMBDA_TARGET="${LAMBDA_TARGET:-1.0}"
LAMBDA_REPEL="${LAMBDA_REPEL:-0.25}"
GT_SEQUENCE_HORIZON="${GT_SEQUENCE_HORIZON:-4}"
GT_SEQUENCE_BANK_PATH="${GT_SEQUENCE_BANK_PATH:-auto}"
RESUME_PATCH_PATH="${RESUME_PATCH_PATH:-}"
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
    --badseq_alpha)
      BADSEQ_ALPHA="$2"
      shift 2
      ;;
    --lambda_target)
      LAMBDA_TARGET="$2"
      shift 2
      ;;
    --lambda_repel)
      LAMBDA_REPEL="$2"
      shift 2
      ;;
    --gt_sequence_horizon)
      GT_SEQUENCE_HORIZON="$2"
      shift 2
      ;;
    --gt_sequence_bank_path)
      GT_SEQUENCE_BANK_PATH="$2"
      shift 2
      ;;
    --resume_patch_path)
      RESUME_PATCH_PATH="$2"
      shift 2
      ;;
    --tags)
      TAGS_CSV="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--iter N] [--innerLoop N] [--lambda_siglip V] [--badseq_alpha V] [--lambda_target V] [--lambda_repel V] [--gt_sequence_horizon N] [--gt_sequence_bank_path PATH|auto] [--resume_patch_path PATH] [--tags tag1,tag2,...]"
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
  TAGS=("offline_contact" "phase1R4" "innerLoop${INNER_LOOP}" "bad_sequence")
fi

echo "[run_offline_sanity_pass_bad_sequence] ITER=${ITER} INNER_LOOP=${INNER_LOOP} LAMBDA_SIGLIP=${LAMBDA_SIGLIP} BADSEQ_ALPHA=${BADSEQ_ALPHA} TAGS=${TAGS[*]}"

python VLAAttacker/UADA_rollout_wrapper.py \
  --dataset libero_spatial \
  --server /home/ubuntu/yxx/projectAttack \
  --device 4 \
  --bs 8 \
  --iter "${ITER}" \
  --lr 5e-3 \
  --warmup 100 \
  --accumulate 1 \
  --innerLoop "${INNER_LOOP}" \
  --attack_mode projection \
  --patch_size 3,80,80 \
  --projection_size 3,80,80 \
  --projection_alpha 0.8 \
  --projection_alpha_jitter 0.02 \
  --projection_fixed_angle true \
  --projection_shear 0.0 \
  --projection_scale_min 0.95 \
  --projection_scale_max 1.05 \
  --projection_keystone_jitter 0.005 \
  --projection_randomization_enabled true \
  --geometry false \
  --phase1_ratio 0.6 \
  --phase1_rollout 4 \
  --phase2_rollout 8 \
  --action_gap_mode bad_sequence \
  --badseq_alpha "${BADSEQ_ALPHA}" \
  --lambda_target "${LAMBDA_TARGET}" \
  --lambda_repel "${LAMBDA_REPEL}" \
  --gt_sequence_horizon "${GT_SEQUENCE_HORIZON}" \
  --gt_sequence_bank_path "${GT_SEQUENCE_BANK_PATH}" \
  --resume_patch_path "${RESUME_PATCH_PATH}" \
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
  --eval_visual_only false \
  --val_max_batches 16 \
  --use_all_joints false \
  --maskidx 0,1,2 \
  --save_interval 100 \
  --sanity_mode false \
  --sanity_num_batches 4 \
  --sanity_disable_randomization false \
  --wandb_entity "1473195970-beihang-university" \
  --wandb_project "projectAttack_bad_sequence" \
  --tags "${TAGS[@]}"
