#!/bin/bash

#SBATCH --job-name GRID_Finetuning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w aurora-g7
#SBATCH -o logs/slurm-%A-%x.out

python eval_wer_finetuning.py \
  --data_dir /local_datasets/GRID_srt/data \
  --speaker s10_processed \
  --checkpoint AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth \
  --lora_checkpoint ./checkpoint_finetuned/epoch=1.ckpt \
  --pretrain_avhubert_enc_video_path av_hubert/checkpoints/large_lrs3_iter5.pt \
  --output_csv result_s10_lora.csv \
  --use_uadf \
  --low-cpu-mem-usage \
  --add_PETF_LLM lora \
  --reduction-lora 64 \
  --alpha 16

# letting slurm know this code finished without any problem
exit 0