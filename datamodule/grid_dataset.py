import os
import glob
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import random

class GRIDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        modality="audiovisual",
        split="train",
        split_ratio=[0.9, 0.1, 0.0], # s1 전용 테스트를 위해 9:1 설정 (Test는 일단 0)
        max_samples=None,
        shuffle=True,
        transform=None, # 외부에서 transform을 받을 수도 있음
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.split = split
        
        # [설정] GRID 데이터셋 전처리 파라미터 (inference_avsr.py 참고)
        self.video_size = 88  # 모델 입력 사이즈
        self.audio_sample_rate = 16000 # Whisper 등 대부분 모델 표준
        
        # 1. 데이터 로드 (경로 문제 해결)
        self.samples = self._load_samples(root_dir)
        
        if len(self.samples) == 0:
             raise RuntimeError(f"No GRID samples found in {root_dir}. Check if 's1_processed' exists.")

        # 2. 셔플 및 분할
        if shuffle:
            random.seed(42)
            random.shuffle(self.samples)
            
        total_len = len(self.samples)
        train_len = int(total_len * split_ratio[0])
        val_len = int(total_len * split_ratio[1])
        # 나머지는 test_len

        if split == "train":
            self.samples = self.samples[:train_len]
        elif split == "val":
            self.samples = self.samples[train_len : train_len + val_len]
        elif split == "test":
            self.samples = self.samples[train_len + val_len :]
            
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples from {root_dir}")

    def _load_samples(self, root_dir):
        """
        /local_datasets/GRID_srt/data 아래의 s1_processed 등을 자동으로 찾음.
        """
        samples = []
        # root_dir 안에 있는 모든 폴더 검색 (s1, s1_processed 등)
        speaker_dirs = glob.glob(os.path.join(root_dir, "s*"))
        
        for spk_dir in speaker_dirs:
            spk_name = os.path.basename(spk_dir)

            if "s10" in spk_name: 
                continue

            align_dir = os.path.join(spk_dir, "align")
            
            if not os.path.exists(align_dir):
                continue    
                
            # .align 파일 찾기
            align_files = glob.glob(os.path.join(align_dir, "*.align"))
            
            for align_path in align_files:
                file_id = os.path.basename(align_path).replace(".align", "")
                
                # 비디오 경로 (.mpg)
                vid_path = os.path.join(spk_dir, f"{file_id}.mpg")
                if not os.path.exists(vid_path):
                    vid_path = os.path.join(spk_dir, f"{file_id}.mp4")
                
                # 파일이 존재하고 정답 텍스트가 있으면 리스트에 추가
                if os.path.exists(vid_path):
                    text = self._load_align(align_path)
                    if text:
                        samples.append({
                            "video_path": vid_path,
                            "audio_path": vid_path, # 오디오도 같은 파일 사용
                            "text": text,
                            "id": file_id,
                            "speaker": spk_name
                        })
        return samples

    def _load_align(self, align_path):
        words = []
        try:
            with open(align_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        word = parts[2]
                        if word not in ["sil", "sp"]:
                            words.append(word)
            return " ".join(words).lower()
        except:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Video Loading & Transform
        # inference_avsr.py의 로직을 참고하여 텐서로 변환
        video = self.load_video(sample["video_path"])
        
        # 2. Audio Loading & Transform
        # .mpg에서 직접 오디오 로드 및 리샘플링
        audio = self.load_audio(sample["audio_path"])

        return {
            "video": video,     # (T, C, H, W) Tensor
            "audio": audio,     # (T_audio,) Tensor
            "text": sample["text"], # Raw Text (Tokenizer는 train.py에서 처리)
            "tokens": sample["text"], # 호환성을 위해 키 추가
            "id": sample["id"]
        }

    def load_video(self, path):
        """
        비디오를 로드하고 (T, H, W, C) -> (T, C, H, W), Grayscale, Resize 수행
        """
        # pts_unit='sec'로 로드
        video, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
        
        # 정규화 (0~255 -> 0~1)
        video = video.float() / 255.0
        
        # Grayscale 변환 (AV-Hubert 등은 보통 흑백을 씁니다. 컬러면 이 줄 주석 처리)
        video = F.rgb_to_grayscale(video)
        
        # Resize (88x88) - inference_avsr.py 참고
        video = F.resize(video, (self.video_size, self.video_size))
        
        # (T, C, H, W) 형태 확인
        return video

    def load_audio(self, path):
        """
        .mpg에서 오디오 로드 후 16kHz로 리샘플링 및 모노 변환
        """
        try:
            waveform, sample_rate = torchaudio.load(path)
            
            # 스테레오 -> 모노 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 리샘플링 (원본 SR -> 16000Hz)
            if sample_rate != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.audio_sample_rate)
                waveform = resampler(waveform)
                
            return waveform.squeeze() # (T,) 형태로 반환
            
        except Exception as e:
            print(f"Error loading audio from {path}: {e}")
            # 에러 시 빈 텐서 혹은 0으로 채운 텐서 반환 (학습 중단 방지)
            return torch.zeros(16000)