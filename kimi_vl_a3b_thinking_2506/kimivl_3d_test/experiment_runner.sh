#!/bin/bash

# ======= ./experiment_runner.sh ========
# For reproducibility and organizational purposes, this script is to be used for listing the experiments we want to run.
#source ~/.bashrc
#pyenv activate kimivl

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {0}"
  exit 1
fi

KIMIVL="${PWD%%LLaVA-3D*}/LLaVA-3D/kimivl/"

MODEL="moonshotai/Kimi-VL-A3B-Thinking-2506"
SCENES="/project/def-wangcs/indrisch/vllm/data/ScanNet/scans"
EXP_DIR="/project/def-wangcs/indrisch/vllm/kimi_vl_a3b_thinking_2506/kimivl_3d_test/experiments"
ANNO_DIR="/project/def-wangcs/indrisch/vllm/data/sqa-3d/ScanQA_format"


# python kimivl_3d_test.py \
#     --question_file ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d.json \
#     --answer_file ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d_answers.json \
#     --image_folder ${SCENES} \
#     --export_json ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers.json \
#     --model_path moonshotai/Kimi-VL-A3B-Thinking \
#     --device cuda \
#     --sample_rate 800

case "$1" in
    0)
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export VLLM_LOGGING_LEVEL=DEBUG
        python kimivl_3d_test.py \
            --question_file ${ANNO_DIR}/SQA_scene0307_00_formatted_LLaVa3d.json \
            --answer_file ${ANNO_DIR}/SQA_scene0307_00_formatted_LLaVa3d_answers.json \
            --image_folder ${SCENES} \
            --export_json ${EXP_DIR}/SQA3D/scene0307_00/SQA_scene0307_00_formatted_LLaVa3d_pred-answers.json \
            --model_path moonshotai/Kimi-VL-A3B-Thinking-2506 \
            --device cuda:0 \
            --sample_rate 800 \
            --num_chunks 1 \
            --chunk_idx 0
        ;;
    *)
        echo "Error: invalid option '$1'. Use 0 or 1."
        exit 2
        ;;
esac  