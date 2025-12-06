#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.lightning import ModelModule_LLM


########################################
# 1. Dataset
########################################

class AVSRFinetuneDataset(Dataset):
    def __init__(self, manifest_path, tokenizer):
        self.items = []
        self.tokenizer = tokenizer

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.items.append(r)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]

        # ===== Audio =====
        import torchaudio
        wav, sr = torchaudio.load(item["audio_path"])
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.transpose(1, 0).float()

        # ===== Video =====
        from datamodule.av_dataset import load_video
        video = load_video(item["video_path"])

        # numpy → torch
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.copy())
        else:
            video = video.clone()

        # squeeze
        while video.ndim > 4:
            video = video.squeeze(0)

        # RGB → grayscale
        if video.ndim == 4 and video.shape[-1] == 3:
            video = (
                0.2989 * video[..., 0] +
                0.5870 * video[..., 1] +
                0.1140 * video[..., 2]
            )

        if video.ndim == 4 and video.shape[-1] == 1:
            video = video[..., 0]

        if video.ndim == 3:
            video = video.unsqueeze(1)

        video = video.float()

        # ===== Text =====
        token_ids = self.tokenizer(
            item["text"],
            return_tensors="pt",
            add_special_tokens=True
        )["input_ids"].squeeze(0)

        return {
            "audio": wav,
            "video": video,
            "tokens": token_ids
        }


########################################
# 2. Collate
########################################

def collate(batch):
    audios = [b["audio"] for b in batch]
    videos = [b["video"] for b in batch]
    tokens = [b["tokens"] for b in batch]

    max_T = max(v.shape[0] for v in videos)
    padded_videos = []
    for v in videos:
        T, C, H, W = v.shape
        pad_T = max_T - T
        if pad_T > 0:
            pad = torch.zeros(pad_T, C, H, W, dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        padded_videos.append(v)

    return {
        "audio": pad_sequence(audios, batch_first=True),
        "video": torch.stack(padded_videos, dim=0),
        "tokens": pad_sequence(tokens, batch_first=True),
        "lengths": torch.tensor([a.size(0) for a in audios])
    }


########################################
# 3. Freeze Backbone
########################################

def freeze_backbone(model):
    if hasattr(model, "audio_encoder") and model.audio_encoder is not None:
        for p in model.audio_encoder.parameters():
            p.requires_grad = False

    if hasattr(model, "video_encoder") and model.video_encoder is not None:
        for p in model.video_encoder.parameters():
            p.requires_grad = False

    if hasattr(model, "llm") and model.llm is not None:
        for p in model.llm.parameters():
            p.requires_grad = False


########################################
# 4. Trainable Params (Projector + LoRA)
########################################

def collect_trainable(model):
    train_params = []
    for name, p in model.named_parameters():
        if ("proj" in name) or ("lora" in name):
            p.requires_grad = True
            train_params.append(p)
        else:
            p.requires_grad = False

    total = sum(p.numel() for p in train_params) / 1e6
    print(f"Trainable parameters = {total:.2f}M")
    return train_params


########################################
# 5. Train loop
########################################

def train_epoch(modelmodule, dataloader, optimizer, device):
    modelmodule.train()
    model = modelmodule.model

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):

        # ========== DEVICE TRANSFER ==========
        batch["audio"] = batch["audio"].to(device).float()
        batch["video"] = batch["video"].to(device).float()
        batch["tokens"] = batch["tokens"].to(device)     # ★ 필수 ★
        batch["lengths"] = batch["lengths"].to(device)

        # projector는 float32로 통일
        for name, module in model.named_modules():
            if "audio_proj" in name or "video_proj" in name:
                module.to(dtype=torch.float32)

        outputs = model(batch, is_trainval=True)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


########################################
# 6. Main
########################################

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--llm-model", type=str, required=True)

    parser.add_argument("--pretrain-avhubert-enc-video-path", type=str, required=True)
    parser.add_argument("--pretrain-avhubert-enc-audio-path", type=str, default=None)
    parser.add_argument("--pretrain-avhubert-enc-audiovisual-path", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--use-uadf", action="store_true")
    parser.add_argument("--uadf-fusion-method", type=str, default="uncertainty")
    parser.add_argument("--uadf-temperature", type=float, default=1.0)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from argparse import Namespace
    dummy_args = Namespace(
        pretrained_model_path=args.pretrained,
        modality="audiovisual",
        llm_model=args.llm_model,

        pretrain_avhubert_enc_video_path=args.pretrain_avhubert_enc_video_path,
        pretrain_avhubert_enc_audio_path=args.pretrain_avhubert_enc_audio_path,
        pretrain_avhubert_enc_audiovisual_path=args.pretrain_avhubert_enc_audiovisual_path,

        audio_encoder_name="openai/whisper-medium.en",

        prompt_audio="Transcribe speech to text.",
        prompt_video="Transcribe video to text.",
        prompt_audiovisual="Transcribe speech and video to text.",

        downsample_ratio_audio=4,
        downsample_ratio_video=2,
        downsample_ratio_audiovisual=3,

        intermediate_size=2048,

        single_projector_avhubert=False,
        use_lora_avhubert=False,
        unfrozen_modules=[None],
        max_dec_tokens=32,
        num_beams=1,

        reduction_lora=None,
        alpha=None,

        use_uadf=args.use_uadf,
        uadf_fusion_method=args.uadf_fusion_method,
        uadf_temperature=args.uadf_temperature,

        tokenizer_path=args.llm_model,
        weight_decay=0.1,
        lr=args.lr,
    )

    modelmodule = ModelModule_LLM(dummy_args)
    modelmodule.to(device)
    model = modelmodule.model

    # projector FP32 통일
    for name, module in model.named_modules():
        if "audio_proj" in name or "video_proj" in name:
            module.float()

    freeze_backbone(model)

    train_params = collect_trainable(model)
    optimizer = torch.optim.AdamW(train_params, lr=args.lr)

    # Dataset
    dataset = AVSRFinetuneDataset(args.manifest, modelmodule.tokenizer)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=collate)

    # Train
    for epoch in range(args.epochs):
        loss = train_epoch(modelmodule, loader, optimizer, device)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

        ckpt = f"finetuned_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
