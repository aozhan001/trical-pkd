#!/bin/bash

# custom config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA="${REPO_ROOT}/promptkd-data"
export PYTHONUTF8=1
TRAINER=TriCal

DATASET=$1 # 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101' 'caltech101'
SEED=$2
GPU_ID=$3
MTP_ALPHA=$4
DVP_ALPHA=$5
PRIOR_GAMMA=$6
MTP_ENABLE=${7:-True}
DVP_ENABLE=${8:-True}
PRIOR_ENABLE=${9:-True}
TARGET_RATIO=${10:-1.0}
# SHOTS=${11:-16}

CFG=vit_b16_c2_ep20_batch8_4+4ctx_cross_datasets
SHOTS=0

DIR=output/xd/${DATASET}/${DATASET}_${MTP_ENABLE}_${DVP_ENABLE}_${PRIOR_ENABLE}/${TRAINER}/${CFG}_${SHOTS}shots/seed_${SEED}

if [[ "${TARGET_RATIO}" != "1" && "${TARGET_RATIO}" != "1.0" ]]; then
    DIR=${DIR}/ratio_${TARGET_RATIO}
fi


CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.TRICAL.TEMPERATURE 1.0 \
    TRAINER.TRICAL.KD_WEIGHT 1000.0 \
    TRAINER.TRICAL.USE_MULTI_TEMPLATE_TEXT ${MTP_ENABLE} \
    TRAINER.TRICAL.DVP_ENABLE ${DVP_ENABLE} \
    TRAINER.TRICAL.PRIOR_CORRECT ${PRIOR_ENABLE} \
    TRAINER.TRICAL.MTP_ALPHA ${MTP_ALPHA} \
    TRAINER.TRICAL.DVP_ALPHA ${DVP_ALPHA} \
    TRAINER.TRICAL.PRIOR_GAMMA ${PRIOR_GAMMA} \
    DATASET.TARGET_DATA_RATIO ${TARGET_RATIO} \
    TRAINER.MODAL cross
