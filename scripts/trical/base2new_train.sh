#!/bin/bash

# custom config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA="${REPO_ROOT}/promptkd-data"
export PYTHONUTF8=1
TRAINER=TriCal

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2
GPU_ID=$3
MTP_ALPHA=${4:-0.2}
DVP_ALPHA=${5:-0.2}
PRIOR_GAMMA=${6:-0.25}
MTP_ENABLE=${7:-True}
DVP_ENABLE=${8:-True}
PRIOR_ENABLE=${9:-True}
SHOTS=${10:-0}
KD_WEIGHT=${11:-}
# SHOTS=${12:-64}



CFG=vit_b16_c2_ep20_batch8_4+4ctx
# SHOTS=0

# DIR=output/base2new/train_base_消融/${DATASET}/${DATASET}_${MTP_ENABLE}_${DVP_ENABLE}_${PRIOR_ENABLE}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
DIR=output/base2new/train_base_promptkd/${DATASET}/${DATASET}_${MTP_ENABLE}_${DVP_ENABLE}_${PRIOR_ENABLE}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
# DIR=output/base2new/train_base/${DATASET}_不预训练/${DATASET}_${MTP_ENABLE}_${DVP_ENABLE}_${PRIOR_ENABLE}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000

if [ -z "${KD_WEIGHT}" ]; then
    case "${DATASET}" in
        fgvc_aircraft|oxford_flowers|dtd)
            KD_WEIGHT=200.0
            ;;
        imagenet|caltech101|eurosat|food101|oxford_pets|stanford_cars|sun397|ucf101)
            KD_WEIGHT=1000.0
            ;;
        *)
            KD_WEIGHT=1000.0
            ;;
    esac
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.TRICAL.TEMPERATURE 1.0 \
    TRAINER.TRICAL.KD_WEIGHT ${KD_WEIGHT} \
    TRAINER.TRICAL.USE_MULTI_TEMPLATE_TEXT ${MTP_ENABLE} \
    TRAINER.TRICAL.DVP_ENABLE ${DVP_ENABLE} \
    TRAINER.TRICAL.PRIOR_CORRECT ${PRIOR_ENABLE}\
    TRAINER.TRICAL.MTP_ALPHA ${MTP_ALPHA}\
    TRAINER.TRICAL.DVP_ALPHA ${DVP_ALPHA}\
    TRAINER.TRICAL.PRIOR_GAMMA ${PRIOR_GAMMA}

    
