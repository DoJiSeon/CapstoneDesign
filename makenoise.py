import os
import glob
import torch
import torchaudio
import math
from tqdm import tqdm

# ================= [설정] =================
# 1. 깨끗한 원본 데이터 경로 (파일 하나만 지정)
CLEAN_ROOT = "./tests/swwv9a.mpg" 

# 2. 소음 파일 경로
NOISE_FILE = "./tests/ch01.wav"

# 3. 결과물이 저장될 경로
OUTPUT_ROOT = "./tests"

# 4. 목표 SNR
TARGET_SNR = 0
# ==========================================

def get_signal_power(waveform):
    return waveform.pow(2).mean()

def mix_audio(clean_waveform, noise_waveform, snr_db):
    clean_len = clean_waveform.shape[1]
    noise_len = noise_waveform.shape[1]
    
    if noise_len < clean_len:
        repeat_times = math.ceil(clean_len / noise_len)
        noise_waveform = noise_waveform.repeat(1, repeat_times)
    
    noise_waveform = noise_waveform[:, :clean_len]
    
    clean_power = get_signal_power(clean_waveform)
    noise_power = get_signal_power(noise_waveform)
    
    if noise_power == 0:
        return clean_waveform

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = (target_noise_power / noise_power).sqrt()
    
    noisy_waveform = clean_waveform + (noise_waveform * scale)
    
    max_val = noisy_waveform.abs().max()
    if max_val > 1.0:
        noisy_waveform = noisy_waveform / max_val
        
    return noisy_waveform

def process_single_file(vid_path, noise, target_snr, output_dir):
    """단일 파일 처리 함수"""
    file_id = os.path.basename(vid_path).split('.')[0] # 확장자 제거
    output_wav_path = os.path.join(output_dir, f"{file_id}.wav")
    
    print(f"🔄 Processing: {vid_path}")
    
    try:
        # 1. 원본 오디오 로드
        clean, sr = torchaudio.load(vid_path)
        
        # 2. 리샘플링 (16k)
        if sr != 16000:
            clean = torchaudio.transforms.Resample(sr, 16000)(clean)
        
        # 3. 채널 맞추기 (Mono)
        if clean.shape[0] > 1:
            clean = torch.mean(clean, dim=0, keepdim=True)
        
        # 소음도 모노로
        if noise.shape[0] > 1:
            noise_mono = torch.mean(noise, dim=0, keepdim=True)
        else:
            noise_mono = noise

        # 4. 섞기
        noisy_audio = mix_audio(clean, noise_mono, target_snr)
        
        # 5. 저장
        torchaudio.save(output_wav_path, noisy_audio, 16000)
        print(f"✅ Saved to: {output_wav_path}")
        
    except Exception as e:
        print(f"❌ Error on {file_id}: {e}")

def main():
    print(f"🔊 Noise Generation Started!")
    print(f"   Clean Source: {CLEAN_ROOT}")
    print(f"   Noise Source: {NOISE_FILE}")
    print(f"   Target SNR: {TARGET_SNR} dB")
    
    # 출력 폴더 생성
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 소음 로드 (한 번만)
    noise, sr_noise = torchaudio.load(NOISE_FILE)
    if sr_noise != 16000:
        resampler = torchaudio.transforms.Resample(sr_noise, 16000)
        noise = resampler(noise)
    
    # [수정된 부분] 파일인지 폴더인지 확인하여 분기 처리
    if os.path.isfile(CLEAN_ROOT):
        # 1. 단일 파일 모드
        process_single_file(CLEAN_ROOT, noise, TARGET_SNR, OUTPUT_ROOT)
        
    else:
        # 2. 데이터셋 폴더 모드 (기존 로직)
        all_speakers = glob.glob(os.path.join(CLEAN_ROOT, "s*"))
        all_speakers.sort()
        
        for spk_dir in tqdm(all_speakers, desc="Processing Speakers"):
            spk_name = os.path.basename(spk_dir)
            out_spk_dir = os.path.join(OUTPUT_ROOT, spk_name)
            os.makedirs(out_spk_dir, exist_ok=True)
            
            video_files = glob.glob(os.path.join(spk_dir, "*.mpg"))
            for vid_path in video_files:
                file_id = os.path.basename(vid_path).replace(".mpg", "")
                output_wav_path = os.path.join(out_spk_dir, f"{file_id}.wav")
                
                if os.path.exists(output_wav_path): continue
                
                # 내부 로직은 단일 처리 함수와 동일하므로 복붙하거나 함수 호출 가능
                # 여기선 기존 흐름 유지를 위해 생략하고 process_single_file 호출로 대체 가능
                process_single_file(vid_path, noise, TARGET_SNR, out_spk_dir)

if __name__ == "__main__":
    main()