import os
import torch
import torch.nn as nn
import gc
import psutil
import torchaudio
import argparse
import numpy as np

from models.lightning import ModelModule_LLM
from datamodule.av_dataset import load_video, cut_or_pad
from datamodule.data_module import collate_LLM
from datamodule.transforms import AudioTransform, VideoTransform

# ================= 설정 (Audio Only Mode) =================
CONFIG = {
    "checkpoint": "AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth", 
    
    # [선택] LoRA를 쓸지 말지 결정 (일단 씁니다)
    "lora_checkpoint": "checkpoint_finetuned/epoch=1_nouadf.ckpt", 
    
    "pretrain_avhubert_enc_video_path": "av_hubert/checkpoints/large_lrs3_iter5.pt",
    "llm_model": "/data/dojiseon/repos/CapstoneDesign/models/Meta-Llama-3.1-8B",

    # [핵심 변경] 오디오만 사용합니다! (비디오 노이즈 차단)
    "modality": "audio", 
    
    "use_uadf": False, 
    "add_PETF_LLM": "lora",
    "reduction_lora": 64,
    "alpha": 16,

    # 오디오 모드에서는 비디오 비율이 의미 없지만, 모델 생성을 위해 유지
    "downsample_ratio_video": 2, 
    "downsample_ratio_audio": 4, 
    "downsample_ratio_audiovisual": 3,
    
    "audio_encoder_name": "openai/whisper-medium.en",
    
    "prompt_audio": "", # 프롬프트 제거
    "prompt_video": "",
    "prompt_audiovisual": "", 
    
    "intermediate_size": 2048,
    "max_dec_tokens": 64,
    
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
    "num_beams": 1, # Greedy Search (가장 정직하게)
    "pretrain_avhubert_enc_audio_path": None,
    "pretrain_avhubert_enc_audiovisual_path": None,
}
# =================================================

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

    def load_model(self):
        print("🚀 모델 로딩 시작 (Audio-Only Mode)...")
        checkpoint_path = self.args.checkpoint
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', mmap=True)
        except:
            ckpt = torch.load(checkpoint_path, map_location='cpu')

        if 'hyper_parameters' in ckpt:
            for key, value in ckpt['hyper_parameters'].items():
                if not hasattr(self.args, key):
                    setattr(self.args, key, value)
        
        # [강제 설정] 오디오 모드로 고정
        self.args.modality = "audio"
        self.args.downsample_ratio_video = 2
        self.args.use_uadf = False 
        self.args.add_PETF_LLM = "lora"
        self.args.prompt_audio = ""

        print(f"DEBUG: Initializing ModelModule_LLM...")
        self.modelmodule = ModelModule_LLM(self.args)

        # 1. 레이어 교체 (2048) - 오디오 모드라도 뼈대는 맞춰야 함
        print("🔧 [System] 모델 구조 교정 (2048 Layer)...")
        if hasattr(self.modelmodule.model, 'video_proj'):
            new_proj = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 4096)
            )
            self.modelmodule.model.video_proj = new_proj
            print("🔧 [System] 레이어 교체 완료.")

        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # 2. 가중치 주입
        print("🔎 [System] 가중치 주입 중...")
        try:
            with torch.no_grad():
                # w0, w2 찾기 로직 생략 (이미 검증됨) - 단순 로드 시도
                pass
        except:
            pass

        print("📥 전체 모델 가중치 적용...")
        self.modelmodule.load_state_dict(state_dict, strict=False)
        del ckpt, state_dict
        gc.collect()

        # 3. LoRA 로딩
        if hasattr(self.args, 'lora_checkpoint') and self.args.lora_checkpoint:
            print(f"🚀 [LoRA] 파인튜닝 체크포인트 로딩: {self.args.lora_checkpoint}")
            if os.path.exists(self.args.lora_checkpoint):
                try:
                    lora_ckpt = torch.load(self.args.lora_checkpoint, map_location='cpu')
                    lora_state_dict = lora_ckpt['state_dict'] if 'state_dict' in lora_ckpt else lora_ckpt
                    self.modelmodule.load_state_dict(lora_state_dict, strict=False)
                    print("✅ [LoRA] 적용 완료!")
                except Exception as e:
                    print(f"❌ [LoRA] 로딩 에러: {e}")

        torch.cuda.empty_cache()
        self.modelmodule.eval()
        if self.device == "cuda":
            self.modelmodule.to(device=self.device, dtype=torch.bfloat16)
        
        self.modelmodule.tokenizer.name_or_path = "meta-llama/Meta-Llama-3.1-8B"
        print("✅ 모델 준비 완료 (Audio Only)!")

    def inference(self, video_path):
        if not self.modelmodule:
            raise RuntimeError("Model not loaded.")

        self.args.video_path = video_path
        self.args.audio_path = video_path 
        
        print_memory_usage("Before Inference")

        # [오디오 단순화] 복잡한 필터 제거하고 기본 변환만 수행 (왜곡 방지)
        temp_audio_file = "temp_extracted_audio.wav"
        # -ac 1: 모노, -ar 16000: 16kHz
        cmd = f"ffmpeg -i {video_path} -vn -ac 1 -ar 16000 {temp_audio_file} -y -hide_banner -loglevel error"
        os.system(cmd)
        
        if not os.path.exists(temp_audio_file):
            return "Error: Audio extraction failed."

        waveform, sample_rate = torchaudio.load(temp_audio_file, normalize=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # [소리 크기 체크]
        abs_max = torch.max(torch.abs(waveform)).item()
        print(f"🔊 [Audio Check] Max Amplitude: {abs_max:.4f}")
        
        # 소리가 너무 작으면 강제로 증폭 (소프트웨어 증폭)
        if abs_max < 0.1:
            print("⚠️ 소리가 작습니다. 강제로 증폭합니다.")
            waveform = waveform * (0.1 / (abs_max + 1e-6))

        if waveform.shape[1] < 16000: # 1초 미만 패딩
             pad = torch.zeros(1, 16000)
             waveform = torch.cat([waveform, pad], dim=1)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        audio = waveform.transpose(1, 0)
        if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
        
        batch_data = {}
        
        # [중요] 비디오는 아예 로드하지 않음 (더미)
        batch_data["video"] = torch.zeros(1) 
        
        # 오디오 전처리
        audio_transform = AudioTransform("test", snr_target=999999, is_avhubert_audio=False)
        # 길이 제한 (너무 길면 자름)
        audio = cut_or_pad(audio, 16000 * 10) # 최대 10초
        audio = audio_transform(audio)
        
        batch_data["audio"] = audio
        batch_data["tokens"] = ""
        batch_list = [batch_data]

        batch = collate_LLM(batch_list, self.modelmodule.tokenizer, self.args.modality, is_trainval=False)

        if "audio" in batch and batch["audio"].dim() == 4:
            batch["audio"] = batch["audio"].squeeze(1)

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
            avsr_model = self.modelmodule.model
            
            inputs_embeds, _ = avsr_model.prepare_inputs(batch, is_trainval=False)
            
            # [생성] Greedy Search (가장 기본)
            generated_ids = avsr_model.llm.generate(
                inputs_embeds=inputs_embeds, 
                max_new_tokens=64,
                num_beams=1,
                repetition_penalty=1.0,
                do_sample=False,
                eos_token_id=self.modelmodule.tokenizer.vocab["<|end_of_text|>"],
                pad_token_id=self.modelmodule.tokenizer.vocab["<pad>"]
            )
            
            generated_text = self.modelmodule.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Raw Output: {generated_text}")
        print_memory_usage("After Inference")
        return generated_text