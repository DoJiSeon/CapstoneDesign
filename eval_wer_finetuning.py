import os
import argparse
import torch
import jiwer
from tqdm import tqdm
import pandas as pd
import re
import contextlib

# 모델 래퍼 클래스 및 추론 함수
from models.lightning import ModelModule_LLM 
from inference_avsr import inference_single_file

def parse_eval_args():
    parser = argparse.ArgumentParser(description="GRID Dataset Evaluation Script (Base + LoRA)")
    
    # === [핵심 수정] 체크포인트 인자 분리 ===
    parser.add_argument("--checkpoint", type=str, required=True, help="베이스 모델 (16GB) 경로")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="파인튜닝된 LoRA 체크포인트 (200MB) 경로")
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrain_avhubert_enc_video_path", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="models/Meta-Llama-3.1-8B")
    
    # === CSV 및 스피커 옵션 ===
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--speaker", type=str, default="s1")

    # === 기타 모델/실험 옵션 ===
    parser.add_argument("--use_uadf", action="store_true")
    parser.add_argument("--output_csv", type=str, default="eval_result.csv")
    parser.add_argument("--modality", type=str, default="audiovisual")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--pretrain_avhubert_enc_audio_path", type=str, default=None)
    parser.add_argument("--pretrain_avhubert_enc_audiovisual_path", type=str, default=None)
    parser.add_argument("--audio_encoder_name", type=str, default="openai/whisper-medium.en")
    parser.add_argument("--downsample-ratio-video", type=int, default=2)
    parser.add_argument("--downsample-ratio-audio", type=int, default=4)
    parser.add_argument("--max-dec-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--use-lora-avhubert", action="store_true")
    parser.add_argument("--single-projector-avhubert", action="store_true")
    parser.add_argument("--grid-resample-audio", action="store_true")
    parser.add_argument("--uadf-fusion-method", type=str, default="uncertainty")
    parser.add_argument("--uadf-temperature", type=float, default=1.0)
    parser.add_argument("--prompt-audio", type=str, default="Transcribe speech to text.")
    parser.add_argument("--prompt-video", type=str, default="Transcribe video to text.")
    parser.add_argument("--prompt-audiovisual", type=str, default="Transcribe speech and video to text.")
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--unfrozen-modules", nargs="*", default=["peft_llm"]) 
    parser.add_argument("--add_PETF_LLM", type=str, default="lora")           
    parser.add_argument("--reduction-lora", type=int, default=64)             
    parser.add_argument("--alpha", type=int, default=8)                       
    parser.add_argument("--downsample-ratio-audiovisual", type=int, default=3)
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", default=True)
    parser.add_argument("--load-in-8bit", action="store_true", default=False) 
    parser.add_argument("--cpu-offload", action="store_true")
    
    # Dummy args for ModelModule init
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="grid")
    parser.add_argument("--exp-dir", type=str, default="")
    parser.add_argument("--project-wandb", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num-check-save", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-num-buckets", type=int, default=400)
    parser.add_argument("--num-average-epochs", type=int, default=1)
    parser.add_argument("--decode-snr-target", type=float, default=9999)
    parser.add_argument("--auto-test", action="store_true")
    parser.add_argument("--slurm-job-id", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None)
    
    return parser.parse_args()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    mapping = {'0': ' zero', '1': ' one', '2': ' two', '3': ' three', '4': ' four',
               '5': ' five', '6': ' six', '7': ' seven', '8': ' eight', '9': ' nine'}
    for k, v in mapping.items(): text = text.replace(k, v)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_checkpoint_weights(model, ckpt_path, desc="Model"):
    """PyTorch Lightning 체크포인트에서 state_dict만 추출하여 로드하는 헬퍼 함수"""
    print(f"📂 Loading {desc}: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 'model.' 접두사 제거 (Lightning Wrapper 때문에 생김)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[6:] # 'model.' 6글자 제거
        else:
            new_key = k
        new_state_dict[new_key] = v
        
    # strict=False: 서로 없는 키는 무시하고 로드 (Base <-> LoRA 병합의 핵심)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"   └─ 결과: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

def main():
    args = parse_eval_args()
    
    print("🚀 모델 초기화 중 (Base Architecture + LoRA Layers)...")
    # ModelModule_LLM이 args를 보고 LoRA 레이어가 포함된 구조를 생성함
    model_module = ModelModule_LLM(args)
    model = model_module.model # 내부 모델 추출
    
    # ================= [1단계] 베이스 모델 (16GB) 로드 =================
    # LoRA 레이어는 초기화 상태로 남고, 베이스 웨이트만 채워짐
    if args.checkpoint:
        load_checkpoint_weights(model, args.checkpoint, desc="Base Model (16GB)")
        
    # ================= [2단계] LoRA 체크포인트 (200MB) 로드 =================
    # 베이스 웨이트는 건드리지 않고, LoRA 레이어 웨이트만 덮어씀
    if args.lora_checkpoint:
        load_checkpoint_weights(model, args.lora_checkpoint, desc="LoRA Adapter (200MB)")
    else:
        print("⚠️ 경고: LoRA 체크포인트가 입력되지 않았습니다! (베이스 모델로만 추론합니다)")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 베이스 ckpt가 bfloat16 이라면:
    target_dtype = torch.bfloat16

    # (훈련을 fp16으로 했다면 torch.float16으로 바꿔도 됨)
    model_module.to(device=device, dtype=target_dtype)
    model_module.eval()

    # 데이터 로드 (CSV or Folder)
    data_list = []
    if args.test_csv:
        print(f"📄 CSV 테스트 모드: {args.test_csv}")
        df = pd.read_csv(args.test_csv)
        for idx, row in df.iterrows():
            video_p = row['video_path']
            # 경로 유연성 확보
            if not os.path.exists(video_p):
                alt = os.path.join(args.data_dir, video_p)
                alt_2 = os.path.join(args.data_dir, video_p.replace("data/", "", 1))
                if os.path.exists(alt): video_p = alt
                elif os.path.exists(alt_2): video_p = alt_2
            
            data_list.append({"video_path": video_p, "text": row['text'], "id": os.path.basename(video_p).split('.')[0]})
    else:
        # 폴더 스캔 모드
        target_speaker_folder = args.speaker
        video_dir = os.path.join(args.data_dir, target_speaker_folder)
        align_dir = os.path.join(video_dir, "align")

        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mpg') or f.endswith('.mp4')]
            # 정렬
            video_files.sort()
            for vid_file in video_files:
                file_id = os.path.splitext(vid_file)[0]
                # align_path = os.path.join(align_dir, file_id + ".align")
                # align은 필수가 아니므로 경로만 추가 (text는 빈칸)
                data_list.append({"video_path": os.path.join(video_dir, vid_file), "text": "", "id": file_id})

    # 평가 루프
    results = []
    total_wer = 0
    count = 0
    
    print(f"[INFO] Evaluating {len(data_list)} samples...")
    
    for item in tqdm(data_list):
        video_path = item["video_path"]
        if not os.path.exists(video_path): continue
        
        args.video_path = video_path
        args.audio_path = video_path
        
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                prediction = inference_single_file(args, model_module).lower().strip()
                
            gt_clean = clean_text(item["text"])
            pred_clean = clean_text(prediction)
            
            if gt_clean:
                wer = jiwer.wer(gt_clean, pred_clean)
            else:
                wer = 0.0 # GT가 없으면 0 처리
                
            results.append({"file": item["id"], "gt": gt_clean, "pred": pred_clean, "wer": wer})
            
            if gt_clean:
                total_wer += wer
                count += 1
            
        except Exception as e:
            print(f"[ERR] {item['id']}: {e}")
            
    if count > 0:
        avg_wer = total_wer / count
        print(f"\n✅ Final Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f"📄 Saved to {args.output_csv}")
    else:
        print("\n⚠️ No valid ground truth found. Just saved predictions.")
        pd.DataFrame(results).to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()