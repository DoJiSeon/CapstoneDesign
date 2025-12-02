#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 파일 AVSR inference 스크립트 (최종: 로그 삭제 + 키 매칭 Fix + OOM 방지)
"""

import sys
import os
import gc 
import torch
import subprocess
import torchaudio
import argparse
import logging
import psutil
import numpy as np
from python_speech_features import logfbank

# HuggingFace 경고 메시지 끄기
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# 모델 관련 모듈 임포트
from models.lightning import ModelModule_LLM
from datamodule.av_dataset import load_video, cut_or_pad
from datamodule.data_module import collate_LLM
from datamodule.transforms import AudioTransform, VideoTransform

# ================= [메모리 확인 함수] =================
def print_memory_usage(step_name=""):
    """현재 CPU 및 GPU 메모리 사용량을 출력합니다."""
    # CPU RAM
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3  # GB 단위
    
    # GPU VRAM
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_msg = f"GPU: {gpu_allocated:.2f}GB (Alloc) / {gpu_reserved:.2f}GB (Res)"
    else:
        gpu_msg = "GPU: N/A"
        
    print(f"[MEMORY] {step_name:<20} | CPU: {cpu_mem:.2f}GB | {gpu_msg}")
# =====================================================

def stacker(feats, stack_order):
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    return feats

def parse_args():
    parser = argparse.ArgumentParser(description="AVSR 단일 파일 inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True, choices=["audio", "video", "audiovisual", "audiovisual_avhubert"])
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--output_txt", type=str, default="inference_result.txt")
    parser.add_argument("--pretrain-avhubert-enc-video-path", type=str, default=None)
    parser.add_argument("--pretrain-avhubert-enc-audio-path", type=str, default=None)
    parser.add_argument("--pretrain-avhubert-enc-audiovisual-path", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--audio-encoder-name", type=str, default=None)
    parser.add_argument("--downsample-ratio-video", type=int, default=2)
    parser.add_argument("--downsample-ratio-audio", type=int, default=4)
    parser.add_argument("--max-dec-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--use-lora-avhubert", action="store_true")
    parser.add_argument("--single-projector-avhubert", action="store_true")
    parser.add_argument("--grid-resample-audio", action="store_true")
    parser.add_argument("--use-uadf", action="store_true")
    parser.add_argument("--uadf-fusion-method", type=str, default="uncertainty")
    parser.add_argument("--uadf-temperature", type=float, default=1.0)
    parser.add_argument("--prompt-audio", type=str, default="Transcribe speech to text.")
    parser.add_argument("--prompt-video", type=str, default="Transcribe video to text.")
    parser.add_argument("--prompt-audiovisual", type=str, default="Transcribe speech and video to text.")
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--unfrozen-modules", nargs="*", default=[None])
    parser.add_argument("--add_PETF_LLM", type=str, default=None)
    parser.add_argument("--reduction-lora", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--downsample-ratio-audiovisual", type=int, default=3)
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", default=True)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()

def load_model_from_checkpoint(checkpoint_path, args):
    """
    로그 최소화 + 스마트 키 매칭 + OOM 방지 로드
    """
    print("LLM & 체크포인트 로딩 시작")
    # print_memory_usage("Init Start")

    # 1. lightning.py 자동 로딩 방지
    temp_path_backup = getattr(args, 'pretrained_model_path', None)
    args.pretrained_model_path = None 

    # 2. 체크포인트 헤더 읽기 (설정 동기화)
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', mmap=True)
    except:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

    # 3. 하이퍼파라미터 동기화
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        for key, value in hparams.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # 4. 필수 기본값 설정
    defaults = {
        "audio_encoder_name": "openai/whisper-medium.en",
        "llm_model": "meta-llama/Meta-Llama-3.1-8B",
        "prompt_audio": "Transcribe speech to text.",
        "prompt_video": "Transcribe video to text.",
        "prompt_audiovisual": "Transcribe speech and video to text.",
        "downsample_ratio_audiovisual": 3,
        "intermediate_size": 2048,
        "modality": args.modality
    }
    for key, val in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)

    # 5. 모델 뼈대 생성 (여기서 로그가 좀 나올 수 있음 - 라이브러리 자체 로그)
    modelmodule = ModelModule_LLM(args)
    args.pretrained_model_path = temp_path_backup
    # print_memory_usage("Skeleton Built")

    # 6. 가중치 로드 (스마트 키 매칭)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # [Smart Key Matching] model. prefix 문제 자동 해결
    model_keys = set(modelmodule.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # 1) 그대로 매칭되는지 확인
    intersection = model_keys.intersection(ckpt_keys)
    
    # 2) 만약 매칭률이 낮으면, prefix 수정 시도
    if len(intersection) < len(model_keys) * 0.5:
        new_state_dict = {}
        for k, v in state_dict.items():
            # ckpt에 model.이 있고, 코드엔 없으면 제거
            if k.startswith("model.") and k[6:] in model_keys:
                new_state_dict[k[6:]] = v
            # ckpt에 model.이 없고, 코드엔 있으면 추가
            elif "model." + k in model_keys:
                new_state_dict["model." + k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # 로드 실행
    modelmodule.load_state_dict(state_dict, strict=False)
    
    del ckpt
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. 하이퍼파라미터 재동기화 & UADF 처리
    if hasattr(modelmodule, "hparams"):
        for key, value in modelmodule.hparams.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    if hasattr(args, 'use_uadf') and args.use_uadf and args.modality != "audiovisual":
        args.use_uadf = False

    # 8. GPU 이동 및 정밀도 강제 통일 (BFloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelmodule.eval()
    
    if device == "cuda":
        # 복잡한 로그 제거하고 조용히 변환
        target_modules = ["audio_encoder", "video_encoder", "audiovisual_encoder", "audio_proj", "video_proj", "audiovisual_proj", "uadf_block", "llm"]
        for mod_name in target_modules:
            if hasattr(modelmodule.model, mod_name):
                module = getattr(modelmodule.model, mod_name)
                if module is not None:
                    setattr(modelmodule.model, mod_name, module.to(device=device, dtype=torch.bfloat16))
        modelmodule.to(device=device, dtype=torch.bfloat16)

    # 로컬 경로 토크나이저 이름 Fix
    modelmodule.tokenizer.name_or_path = "meta-llama/Meta-Llama-3.1-8B"

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("LLM & 체크포인트 로딩 완료")
    # print_memory_usage("Ready to Infer")
    return modelmodule

import subprocess # [필수] 맨 위에 import subprocess가 없다면 이 함수 안에라도 있어야 함

def inference_single_file(args, modelmodule):
    print("🚀 추론 시작...")
    print_memory_usage("Before Inference")
    
    # Transform 설정
    is_avhubert_audio = True if args.modality == 'audiovisual_avhubert' or (hasattr(args, 'audio_encoder_name') and args.audio_encoder_name == 'av-hubert' and args.modality == 'audio') else False
    video_transform = VideoTransform("test") if args.modality in ["video", "audiovisual", "audiovisual_avhubert"] else None
    audio_transform = AudioTransform("test", snr_target=999999, is_avhubert_audio=is_avhubert_audio) if args.modality in ["audio", "audiovisual", "audiovisual_avhubert"] else None
    
    batch_data = {}
    rate_ratio = 640
    
    # ================= [스마트 오디오 추출 로직 (FFmpeg)] =================
    # 비디오 경로만 있고 오디오 경로가 없으면 비디오 사용
    if args.video_path and not args.audio_path:
        args.audio_path = args.video_path

    # 임시 오디오 파일 경로
    temp_audio_file = "temp_extracted_audio.wav"
    
    if args.modality != "video": # 오디오가 필요한 경우
        if not args.audio_path: 
            raise ValueError("--audio_path required")
            
        # 입력 파일이 비디오 파일인 경우 (확장자로 판단)
        if args.audio_path.lower().endswith(('.mp4', '.mpg', '.avi', '.mov', '.mkv')):
            print(f"ℹ️ [INFO] 비디오 파일 감지: {args.audio_path}")
            print(f"   -> FFmpeg로 오디오 추출 중 (16kHz, Mono)...")
            
            # ffmpeg 명령어로 wav 추출 + 16k 변환 + 모노 변환을 한 번에 수행
            # -y: 덮어쓰기, -vn: 비디오 제외, -ac 1: 모노, -ar 16000: 16k Hz
            cmd = f"ffmpeg -i {args.audio_path} -vn -ac 1 -ar 16000 {temp_audio_file} -y -hide_banner -loglevel error"
            os.system(cmd)
            
            # 타겟 오디오 경로를 임시 파일로 변경
            target_audio_path = temp_audio_file
        else:
            target_audio_path = args.audio_path

        # 1. 오디오 로드 (이제 무조건 wav 파일임)
        waveform, sample_rate = torchaudio.load(target_audio_path, normalize=True)
        
        # 혹시 모르니 안전장치 (이미 ffmpeg가 했겠지만)
        if sample_rate != 16000: 
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 4. Transpose [Time, 1]
        audio = waveform.transpose(1, 0)
        
        # 사용 후 임시 파일 삭제
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

    # =================================================================

    if args.modality == "video":
        if not args.video_path: raise ValueError("--video_path required")
        video = load_video(args.video_path)
        video = video_transform(video)
        if len(video.shape) == 3: video = video.unsqueeze(1)
        if hasattr(args, 'downsample_ratio_video') and args.downsample_ratio_video:
            video = video[: video.size(0) // args.downsample_ratio_video * args.downsample_ratio_video]
        batch_data["video"] = video
        batch_data["tokens"] = ""

    else:
        # 5. 비디오 처리 (Audiovisual)
        if args.modality in ["audiovisual", "audiovisual_avhubert"]:
            video = load_video(args.video_path)
            audio = cut_or_pad(audio, len(video) * rate_ratio)
            video = video_transform(video)
            if len(video.shape) == 3: video = video.unsqueeze(1)
            
            if hasattr(args, 'downsample_ratio_video') and args.downsample_ratio_video:
                if args.modality == "audiovisual_avhubert" and hasattr(args, 'single_projector_avhubert') and args.single_projector_avhubert:
                    pass
                else:
                    video = video[: video.size(0) // args.downsample_ratio_video * args.downsample_ratio_video]
            batch_data["video"] = video

        audio = audio_transform(audio)
        batch_data["audio"] = audio
        batch_data["tokens"] = ""

    batch_list = [batch_data]
    batch = collate_LLM(batch_list, modelmodule.tokenizer, args.modality, is_trainval=False)
    
    if "audio" in batch:
        audio_tensor = batch["audio"]
        if audio_tensor.dim() == 4 and audio_tensor.shape[1] == 1:
            batch["audio"] = audio_tensor.squeeze(1)
    if "video" in batch:
        video_tensor = batch["video"]
        if video_tensor.dim() == 6 and video_tensor.shape[1] == 1:
            batch["video"] = video_tensor.squeeze(1)
    
    device = next(modelmodule.model.parameters()).device
    param_dtype = next(modelmodule.model.parameters()).dtype  # 모델이 쓰는 dtype 그대로
    target_dtype = param_dtype

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            tensor = batch[key]
            if tensor.is_floating_point():
                batch[key] = tensor.to(device=device, dtype=target_dtype)
            else:
                batch[key] = tensor.to(device=device)
    
    modelmodule.eval()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print_memory_usage("Start Generating")
    with torch.inference_mode():
        generated_ids = modelmodule.model(batch, is_trainval=False)
        generated_text = modelmodule.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print_memory_usage("After Inference")
    
    return generated_text

def main():
    args = parse_args()
    modelmodule = load_model_from_checkpoint(args.checkpoint, args)
    predicted_text = inference_single_file(args, modelmodule)
    
    print(f"\n{'='*50}")
    print(f"생성된 텍스트: {predicted_text}")
    print(f"{'='*50}\n")
    
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(predicted_text + "\n")
    print(f"Inference result saved to {args.output_txt}")

if __name__ == "__main__":
    main()