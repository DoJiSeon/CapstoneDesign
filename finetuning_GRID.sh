#!/bin/bash

#SBATCH --job-name GRID_Finetuning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w aurora-g7
#SBATCH -o logs/slurm-%A-%x.out

export WANDB_API_KEY="cb01f61ebb54e749ae4ac404316d2eff8ba76e36"

python train.py \
  --dataset-name grid \
  --modality audiovisual \
  --llm-model ./models/Meta-Llama-3.1-8B \
  --audio-encoder-name openai/whisper-medium.en \
  --root-dir /local_datasets/GRID_srt/data \
  --exp-dir ./checkpoint_finetuned \
  --pretrained-model-path AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth \
  --pretrain-avhubert-enc-video-path av_hubert/checkpoints/large_lrs3_iter5.pt \
  --max-epochs 2 \
  --lr 1e-4 \
  --add_PETF_LLM lora \
  --reduction_lora 64 \
  --alpha 16 \
  --unfrozen_modules peft_llm uadf\
  --downsample-ratio-audio 4 \
  --downsample-ratio-video 2 \
  --accumulate-grad-batches 8 \
  --grid-max-train-samples 10000 \
  --grid-max-val-samples 1500 \
  --use-uadf


# letting slurm know this code finished without any problem
exit 0