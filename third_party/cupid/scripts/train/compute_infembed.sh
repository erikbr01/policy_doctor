#!/bin/bash
# Compute InfEmbed embeddings and save to eval_dir/<trak_exp>/infembed_embeddings.npz.
# Run after TRAK (train_trak.sh) so the same eval_dir and TRAK experiment exist.
#
# Order preservation: rollout embeddings follow episode order (ep0, ep1, ...) from
# eval_dir/episodes. EpisodeDataset sorts files by episode index (numeric). The
# influence visualizer expects this order; a mismatch causes wrong rollout_idx in clusters.
#
# Set the variables below to match your TRAK run (same train_date, eval_date, task,
# state, loss_fn, num_timesteps, featurize_holdout). Set DEBUG=1 to print the
# command without running. Run from repo root or any dir (script cd's to repo root).

set -e

DEBUG=0
OVERWRITE=0
seed=0

train_output_dir="data/outputs/train"
eval_output_dir="data/outputs/eval_save_episodes"
train_date="jan18"
eval_date="jan18"
eval_ckpt="latest"
train_ckpt="latest"
train_seed="0"
eval_seed="0"
eval_as_train_seed=1
device="cuda:0"
loss_fn="square"
num_timesteps=64
batch_size=32
featurize_holdout=1
projection_dim=100
arnoldi_dim=200

# Task / method (must match train_trak.sh)
task="lift_mh"
state="low_dim"
method="diffusion_policy_cnn"
policy="diffusion_unet_lowdim"

declare -A MODEL_KEYS=(
    ["low_dim"]="model."
    ["image"]="obs_encoder.,model."
)
declare -A MODELOUT_FN=(
    ["diffusion_policy_cnn_low_dim"]="DiffusionLowdimFunctionalModelOutput"
    ["diffusion_policy_cnn_image"]="DiffusionHybridImageFunctionalModelOutput"
)
declare -A BATCH_SIZE_MAP=(
    ["lift_mh_low_dim"]=128
    ["lift_mh_image"]=32
)

train_name="${train_date}_train_${policy}_${task}_${train_seed}"
train_dir="${train_output_dir}/${train_date}/${train_name}"
if [[ $eval_as_train_seed == 1 ]]; then
    eval_name="${train_date}_train_${policy}_${task}_${train_seed}"
else
    eval_name="${train_date}_train_${policy}_${task}_${eval_seed}"
fi
eval_dir="${eval_output_dir}/${eval_date}/${eval_name}/${eval_ckpt}"

batch_size="${BATCH_SIZE_MAP[${task}_${state}]:-32}"
modelout_fn="${MODELOUT_FN[${method}_${state}]}"
model_keys="${MODEL_KEYS[${state}]}"

CMD="python compute_infembed_embeddings.py"
CMD="${CMD} --exp_name=auto"
CMD="${CMD} --eval_dir=${eval_dir}"
CMD="${CMD} --train_dir=${train_dir}"
CMD="${CMD} --train_ckpt=${train_ckpt}"
CMD="${CMD} --modelout_fn=${modelout_fn}"
CMD="${CMD} --loss_fn=${loss_fn}"
CMD="${CMD} --num_timesteps=${num_timesteps}"
CMD="${CMD} --batch_size=${batch_size}"
CMD="${CMD} --device=${device}"
CMD="${CMD} --projection_dim=${projection_dim}"
CMD="${CMD} --arnoldi_dim=${arnoldi_dim}"
if [[ -n "${model_keys}" ]]; then
    CMD="${CMD} --model_keys=${model_keys}"
fi
if [[ $featurize_holdout == 1 ]]; then
    CMD="${CMD} --featurize_holdout"
fi
CMD="${CMD} --seed=${seed}"
if [[ $OVERWRITE == 1 ]]; then
    CMD="${CMD} --overwrite"
fi
# Forward any extra args (e.g. --predict_only) to the Python script
CMD="${CMD} $*"

echo "Running: $CMD"
if [[ ${DEBUG} == 0 ]]; then
    cd "$(dirname "$0")/../.." && eval "${CMD}"
else
    echo "(DEBUG=1, not executing)"
fi
