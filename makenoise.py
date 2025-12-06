import os
import subprocess
import shutil
import numpy as np
import librosa
import soundfile as sf

def add_noise_to_video(video_path, noise_path, output_path, target_snr_db=10):
    """비디오 오디오에 노이즈를 섞고 저장하는 함수 (이전과 동일)"""
    temp_audio_path = "temp_clean.wav"
    temp_noisy_audio_path = "temp_noisy.wav"

    try:
        # 1. 오디오 추출
        subprocess.call([
            'ffmpeg', '-y', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
            temp_audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. 로딩 및 노이즈 합성
        clean_sig, sr = librosa.load(temp_audio_path, sr=16000)
        noise_sig, _ = librosa.load(noise_path, sr=16000)

        if len(noise_sig) < len(clean_sig):
            tile_count = int(np.ceil(len(clean_sig) / len(noise_sig)))
            noise_sig = np.tile(noise_sig, tile_count)
        noise_sig = noise_sig[:len(clean_sig)]

        clean_power = np.sum(clean_sig ** 2) / len(clean_sig)
        noise_power = np.sum(noise_sig ** 2) / len(noise_sig)
        
        if noise_power == 0:
            scale = 0
        else:
            target_noise_power = clean_power / (10 ** (target_snr_db / 10))
            scale = np.sqrt(target_noise_power / noise_power)
        
        noisy_sig = clean_sig + (noise_sig * scale)
        
        max_val = np.max(np.abs(noisy_sig))
        if max_val > 1.0: noisy_sig = noisy_sig / max_val

        sf.write(temp_noisy_audio_path, noisy_sig, sr)

        # 3. 병합
        subprocess.call([
            'ffmpeg', '-y', '-i', video_path, '-i', temp_noisy_audio_path,
            '-c:v', 'copy', '-c:a', 'mp2', '-map', '0:v:0', '-map', '1:a:0',
            output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        if os.path.exists(temp_noisy_audio_path): os.remove(temp_noisy_audio_path)

# --- 실행 설정 ---
source_folder = "/local_datasets/GRID_srt/data/s10_processed"       # 원본 데이터 폴더
output_folder = "/local_datasets/GRID_srt/noisedata_DKITCHEN_SNR0"       # 결과 저장 폴더
noise_file = "./tests/ch01.wav"  # 다운받은 노이즈 파일 경로 (변경 필요)
SNR = 0                                 # 테스트할 SNR (0, 5, 10, 15 등)

# 1. 출력 폴더 생성 (없으면 생성)
os.makedirs(output_folder, exist_ok=True)

# 2. 비디오 처리 (mpg 파일만 골라서 노이즈 추가)
files = [f for f in os.listdir(source_folder) if f.endswith('.mpg')]
print(f"총 {len(files)}개의 비디오 파일 처리를 시작합니다. (SNR: {SNR}dB)")

for i, filename in enumerate(files):
    src_video_path = os.path.join(source_folder, filename)
    dst_video_path = os.path.join(output_folder, filename)
    
    add_noise_to_video(src_video_path, noise_file, dst_video_path, target_snr_db=SNR)
    
    if (i+1) % 50 == 0:
        print(f"{i+1}/{len(files)} 비디오 처리 완료...")

# 3. Align 폴더 통째로 복사 (이 부분이 변경되었습니다)
src_align_dir = os.path.join(source_folder, "align")
dst_align_dir = os.path.join(output_folder, "align")

if os.path.exists(src_align_dir):
    print("Align 폴더 복사를 시작합니다...")
    # dirs_exist_ok=True: 이미 폴더가 있어도 덮어쓰거나 병합함 (Python 3.8+)
    shutil.copytree(src_align_dir, dst_align_dir, dirs_exist_ok=True)
    print("Align 폴더 복사 완료.")
else:
    print(f"경고: 원본 경로에 align 폴더가 없습니다. ({src_align_dir})")

print("모든 작업이 완료되었습니다.")