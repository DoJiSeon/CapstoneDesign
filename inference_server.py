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
    "checkpoint": "AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth", 
    
    # [수정] 두 개의 LoRA 경로를 모두 등록
    "lora_uadf": "epoch=1_uadf.ckpt",      # UADF 사용 체크포인트
    "lora_nouadf": "epoch=1_nouadf.ckpt",  # UADF 미사용 체크포인트

    "pretrain_avhubert_enc_video_path": "av_hubert/checkpoints/large_lrs3_iter5.pt",
    "llm_model": "/data/dojiseon/repos/CapstoneDesign/models/Meta-Llama-3.1-8B",
    
    "modality": "audiovisual", 
    "add_PETF_LLM": "lora",
    "reduction_lora": 64,
    "alpha": 8,
    
    # 나머지 설정 유지
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
    
    "use_uadf": False # 초기값
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
        self.current_mode = None  # 현재 로드된 모드 추적 ('uadf' or 'nouadf')

    def _load_weights(self, model, ckpt_path, desc):
        print(f"📂 Loading {desc}: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
        except:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        model_keys = set(model.state_dict().keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.") and k[6:] in model_keys:
                new_state_dict[k[6:]] = v
            elif "model." + k in model_keys:
                new_state_dict["model." + k] = v
            else:
                new_state_dict[k] = v
        
        # LoRA 교체 시 Base Weight는 건드리지 않으므로 strict=False 필수
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        # print(f"   └─ 결과: Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        del ckpt, state_dict
        gc.collect()

    def load_model(self):
        print("🚀 모델 로딩 시작 (Base Model Initializing)...")
        
        # [중요] UADF 레이어를 메모리에 할당하기 위해 초기화는 무조건 True로 시작하거나
        # 구조가 가장 큰 쪽(UADF 사용)에 맞춰서 모델을 생성해 둬야 안전합니다.
        # 여기서는 args.use_uadf 상태에 따라 forward가 바뀌므로, 모델 구조 자체는 UADF 포함으로 생성합니다.
        self.args.use_uadf = True 
        
        self.modelmodule = ModelModule_LLM(self.args)

        # 1. Base Model 로드 (시간 오래 걸림, 딱 한 번만 실행)
        if self.args.checkpoint:
            self._load_weights(self.modelmodule, self.args.checkpoint, "Base Model")

        # 2. 초기 상태는 No-UADF로 설정 (가볍게 시작)
        self.switch_lora_mode(use_uadf=False)

        torch.cuda.empty_cache()
        self.modelmodule.eval()
        if self.device == "cuda":
            self.modelmodule.to(device=self.device, dtype=torch.bfloat16)
        
        self.modelmodule.tokenizer.name_or_path = "meta-llama/Meta-Llama-3.1-8B"
        print("✅ 모델 로딩 및 초기화 완료!")

    def switch_lora_mode(self, use_uadf: bool):
        """
        [핵심 기능] 요청된 모드에 따라 LoRA 가중치만 쏙 갈아끼우는 함수
        """
        target_mode = 'uadf' if use_uadf else 'nouadf'
        
        # 이미 해당 모드라면 스킵 (불필요한 로딩 방지)
        if self.current_mode == target_mode:
            return

        print(f"🔄 모드 전환 중: {self.current_mode} -> {target_mode} ...")
        
        # 1. args 업데이트 (Forward 로직 변경용)
        self.args.use_uadf = use_uadf
        # 모델 내부 hparams도 동기화 (Lightning Module 특성상)
        if hasattr(self.modelmodule, 'hparams'):
             self.modelmodule.hparams.use_uadf = use_uadf

        # 2. 체크포인트 경로 선택
        ckpt_path = CONFIG['lora_uadf'] if use_uadf else CONFIG['lora_nouadf']
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"⚠️ 경고: {target_mode}용 체크포인트 파일이 없습니다! ({ckpt_path})")
            return

        # 3. 가중치 교체 (Base 모델은 그대로, LoRA 부분만 덮어씌움)
        self._load_weights(self.modelmodule, ckpt_path, f"LoRA ({target_mode})")
        
        # 4. 상태 업데이트
        self.current_mode = target_mode
        
        # GPU 메모리 정리 (파편화 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"✅ 모드 전환 완료! 현재 상태: UADF={'ON' if use_uadf else 'OFF'}")

    def process_video_for_inference(self, input_path):
        random_suffix = str(uuid.uuid4())[:8]
        processed_path = f"processed_{random_suffix}.mp4"
        cmd = f'ffmpeg -i "{input_path}" -vf "fps=25" -ar 16000 -ac 1 "{processed_path}" -y -hide_banner -loglevel error'
        print(f"🔄 영상 전처리 중 (25fps 변환): {input_path} -> {processed_path}")
        exit_code = os.system(cmd)
        if exit_code != 0: return None
        return processed_path

    def inference(self, video_path, use_uadf=False):
        """
        inference 함수가 이제 use_uadf 파라미터를 받습니다.
        """
        if not self.modelmodule:
            raise RuntimeError("Model not loaded.")
        
        # [핵심] 추론 시작 전에 모드 확인 및 전환
        self.switch_lora_mode(use_uadf)
        
        print_memory_usage("Before Inference")

        processed_path = self.process_video_for_inference(video_path)
        if not processed_path or not os.path.exists(processed_path):
            return "Error: Video processing failed."

        self.args.video_path = processed_path
        self.args.audio_path = processed_path 
        
        try:
            waveform, sample_rate = torchaudio.load(processed_path, normalize=True)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        except Exception as e:
            if os.path.exists(processed_path): os.remove(processed_path)
            return "Error: Could not load audio."

        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
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
            video = video[: video.size(0) // self.args.downsample_ratio_video * self.args.downsample_ratio_video]
        audio = audio_transform(audio)
        
        batch_data = {"video": video, "audio": audio, "tokens": ""}
        batch = collate_LLM([batch_data], self.modelmodule.tokenizer, self.args.modality, is_trainval=False)

        if "audio" in batch and batch["audio"].dim() == 4: batch["audio"] = batch["audio"].squeeze(1)
        if "video" in batch and batch["video"].dim() == 6: batch["video"] = batch["video"].squeeze(1)

        target_dtype = torch.bfloat16
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                t = batch[key]
                if t.is_floating_point(): batch[key] = t.to(device=self.device, dtype=target_dtype)
                else: batch[key] = t.to(device=self.device)

        print("generating...")
        with torch.inference_mode():
            generated_ids = self.modelmodule.model(batch, is_trainval=False)
            generated_text = self.modelmodule.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"\n{'='*20} [Inference Result (UADF={use_uadf})] {'='*20}")
        print(f"▶ Generated Sentence: {generated_text}")
        print(f"{'='*60}\n")
        
        print_memory_usage("After Inference")
        return generated_text