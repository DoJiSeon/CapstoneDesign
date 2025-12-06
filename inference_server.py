import os
import torch
import torch.nn as nn
import gc
import psutil
import torchaudio
import argparse
import numpy as np
import uuid

from models.lightning import ModelModule_LLM
from datamodule.av_dataset import load_video, cut_or_pad
from datamodule.data_module import collate_LLM
from datamodule.transforms import AudioTransform, VideoTransform

# ================= 설정 (CONFIG) =================
CONFIG = {
    # [1] 베이스 모델 (16GB 짜리 큰 파일)
    "checkpoint": "AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth", 
    
    # [2] LoRA 체크포인트 (사용자가 학습한 작은 파일, 예: epoch=5.ckpt)
    # 없다면 None으로 두세요 (베이스 모델만 사용됨)
    "lora_checkpoint": "./checkpoint_finetuned/epoch=1_uadf.ckpt",  # <--- 여기에 LoRA 파일 경로를 넣으세요! (예: "checkpoints/epoch=10.ckpt")

    "pretrain_avhubert_enc_video_path": "av_hubert/checkpoints/large_lrs3_iter5.pt",
    "llm_model": "./models/Meta-Llama-3.1-8B",
    
    "modality": "audiovisual", 
    "use_uadf": True,
    
    # [3] LoRA 활성화 설정 (LoRA를 쓴다면 "lora"로 설정 필수)
    "add_PETF_LLM": "lora",   # None -> "lora" 로 변경 (LoRA 구조 생성용)
    "reduction_lora": 64,     # 학습할 때 썼던 랭크 (보통 64, 16 등)
    "alpha": 8,               # 학습할 때 썼던 알파 (보통 8)
    
    "downsample_ratio_video": 2, 
    "downsample_ratio_audio": 4,
    "downsample_ratio_audiovisual": 3,
    "audio_encoder_name": "openai/whisper-medium.en",
    "prompt_audio": "Transcribe speech to text.",
    "prompt_video": "Transcribe video to text.",
    "prompt_audiovisual": "Transcribe speech and video to text.",
    "intermediate_size": 2048,
    "max_dec_tokens": 32,
    "num_beams": 1,
    "use_lora_avhubert": False,
    "single_projector_avhubert": False,
    "grid_resample_audio": False,
    "uadf_fusion_method": "uncertainty",
    "uadf_temperature": 1.0,
    "unfrozen_modules": [None],
    "pretrained_model_path": None,
    "use_half_precision": False,
    "low_cpu_mem_usage": True,
    "load_in_8bit": False,
    "cpu_offload": False,
    "pretrain_avhubert_enc_audio_path": None,
    "pretrain_avhubert_enc_audiovisual_path": None,
}

def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_msg = f"GPU: {gpu_allocated:.2f}GB (Alloc)"
    else:
        gpu_msg = "GPU: N/A"
    print(f"[MEMORY] {step_name:<20} | CPU: {cpu_mem:.2f}GB | {gpu_msg}")

class AVSRInferenceHandler:
    def __init__(self):
        self.args = argparse.Namespace(**CONFIG)
        self.modelmodule = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_weights(self, model, ckpt_path, desc):
        """체크포인트 로드 헬퍼 함수"""
        print(f"📂 Loading {desc}: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
        except:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # 스마트 키 매칭 (model. 접두사 처리)
        model_keys = set(model.state_dict().keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.") and k[6:] in model_keys:
                new_state_dict[k[6:]] = v
            elif "model." + k in model_keys:
                new_state_dict["model." + k] = v
            else:
                new_state_dict[k] = v
        
        # strict=False로 로드 (LoRA와 Base 간의 불일치 허용)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"   └─ 결과: Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        del ckpt, state_dict
        gc.collect()

    def load_model(self):
        print("🚀 모델 로딩 시작 (Base + LoRA)...")
        
        # Config 강제 적용
        self.args.downsample_ratio_video = 2
        self.args.use_uadf = False
        
        # 모델 구조 생성 (여기서 LoRA 레이어도 같이 생성됨)
        print(f"DEBUG: Initializing ModelModule_LLM...")
        self.modelmodule = ModelModule_LLM(self.args)

        # 1. 베이스 모델 로드
        if self.args.checkpoint:
            self._load_weights(self.modelmodule, self.args.checkpoint, "Base Model")
        
        # 2. LoRA 체크포인트 로드 (덮어쓰기)
        if hasattr(self.args, 'lora_checkpoint') and self.args.lora_checkpoint:
            self._load_weights(self.modelmodule, self.args.lora_checkpoint, "LoRA Adapter")
        else:
            print("⚠️ [Warning] LoRA 체크포인트가 설정되지 않았습니다! Base 모델로만 동작합니다.")

        torch.cuda.empty_cache()

        self.modelmodule.eval()
        if self.device == "cuda":
            self.modelmodule.to(device=self.device, dtype=torch.bfloat16)
        
        self.modelmodule.tokenizer.name_or_path = "meta-llama/Meta-Llama-3.1-8B"
        print("✅ 모델 로딩 완료!")

    def process_video_for_inference(self, input_path):
        """입력된 영상을 모델이 요구하는 포맷(25fps, 16kHz)으로 강제 변환"""
        random_suffix = str(uuid.uuid4())[:8]
        processed_path = f"processed_{random_suffix}.mp4"
        
        # 싱크 문제 해결을 위한 FPS 25 강제 변환
        cmd = f'ffmpeg -i "{input_path}" -vf "fps=25" -ar 16000 -ac 1 "{processed_path}" -y -hide_banner -loglevel error'
        
        print(f"🔄 영상 전처리 중 (25fps 변환): {input_path} -> {processed_path}")
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print("❌ FFmpeg 변환 실패!")
            return None
        return processed_path

    def inference(self, video_path):
        if not self.modelmodule:
            raise RuntimeError("Model not loaded.")
        
        print_memory_usage("Before Inference")

        # 1. 영상 전처리 (FPS 25 고정 + 오디오 16k)
        processed_path = self.process_video_for_inference(video_path)
        
        if not processed_path or not os.path.exists(processed_path):
            return "Error: Video processing failed."

        # 전처리된 파일 사용
        self.args.video_path = processed_path
        self.args.audio_path = processed_path 
        
        try:
            waveform, sample_rate = torchaudio.load(processed_path, normalize=True)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        except Exception as e:
            print(f"❌ 오디오 로드 실패: {e}")
            if os.path.exists(processed_path): os.remove(processed_path)
            return "Error: Could not load audio."

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if torch.max(torch.abs(waveform)) < 0.001:
            print("⚠️ [Warning] 입력된 오디오가 무음(Silence)입니다!")

        audio = waveform.transpose(1, 0)
        video = load_video(processed_path)
        
        if os.path.exists(processed_path): os.remove(processed_path)

        if len(video) < 2: return "Error: Video too short."

        rate_ratio = 640
        audio = cut_or_pad(audio, len(video) * rate_ratio) 
        
        video_transform = VideoTransform("test")
        audio_transform = AudioTransform("test", snr_target=999999, is_avhubert_audio=False)

        video = video_transform(video)
        if len(video.shape) == 3: video = video.unsqueeze(1)
        
        if hasattr(self.args, 'downsample_ratio_video') and self.args.downsample_ratio_video:
            ratio = self.args.downsample_ratio_video
            video = video[: video.size(0) // ratio * ratio]

        audio = audio_transform(audio)
        
        batch_data = {"video": video, "audio": audio, "tokens": ""}
        batch = collate_LLM([batch_data], self.modelmodule.tokenizer, self.args.modality, is_trainval=False)

        if "audio" in batch and batch["audio"].dim() == 4:
            batch["audio"] = batch["audio"].squeeze(1)
        if "video" in batch and batch["video"].dim() == 6:
            batch["video"] = batch["video"].squeeze(1)

        target_dtype = torch.bfloat16
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                t = batch[key]
                if t.is_floating_point():
                    batch[key] = t.to(device=self.device, dtype=target_dtype)
                else:
                    batch[key] = t.to(device=self.device)

        print("generating...")
        with torch.inference_mode():
            generated_ids = self.modelmodule.model(batch, is_trainval=False)
            generated_text = self.modelmodule.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"\n{'='*20} [Inference Result] {'='*20}")
        print(f"▶ Generated Sentence: {generated_text}")
        print(f"{'='*60}\n")
        
        print_memory_usage("After Inference")
        return generated_text