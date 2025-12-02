import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 설정 =================
# 데이터셋 루트 경로 (사용자 환경에 맞춤)
DATA_ROOT = "/local_datasets/GRID_srt/data"
SPEAKER = "s1_processed"
OUTPUT_TRAIN = "train.csv"
OUTPUT_VALID = "valid.csv"
TEST_SIZE = 0.1  # 10%를 검증용으로 사용 (나머지 90% 학습)
# ========================================

def parse_align_file(align_path):
    """
    .align 파일을 읽어서 문장으로 변환합니다.
    sil(묵음), sp(짧은 멈춤)은 제외합니다.
    """
    words = []
    try:
        with open(align_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2]
                    # sil(묵음), sp(공백) 제외
                    if word not in ["sil", "sp"]:
                        words.append(word)
        return " ".join(words).lower()
    except Exception as e:
        print(f"Error reading {align_path}: {e}")
        return None

def main():
    speaker_dir = os.path.join(DATA_ROOT, SPEAKER)
    align_dir = os.path.join(speaker_dir, "align")
    
    if not os.path.exists(speaker_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {speaker_dir}")
        return

    # .align 파일 목록 가져오기
    align_files = glob.glob(os.path.join(align_dir, "*.align"))
    print(f"🔍 총 {len(align_files)}개의 align 파일을 찾았습니다.")

    data_list = []

    for align_path in tqdm(align_files, desc="데이터 처리 중"):
        file_id = os.path.basename(align_path).replace(".align", "")
        
        # 텍스트 추출
        text = parse_align_file(align_path)
        if not text:
            continue
            
        # 비디오 경로 매칭 (.mpg 파일이 speaker_dir 바로 아래에 있다고 가정)
        video_path = os.path.join(speaker_dir, f"{file_id}.mpg")
        
        # 비디오 파일이 실제로 존재하는지 확인
        if os.path.exists(video_path):
            data_list.append({
                "video_path": video_path,
                "text": text
            })

    # DataFrame 생성
    df = pd.DataFrame(data_list)
    print(f"✅ 유효한 데이터 쌍: {len(df)}개")

    # Train / Valid 나누기 (랜덤 셔플)
    train_df, valid_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)

    # CSV 저장
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    valid_df.to_csv(OUTPUT_VALID, index=False)

    print(f"\n🎉 완료!")
    print(f"📂 학습용 데이터: {OUTPUT_TRAIN} ({len(train_df)}개)")
    print(f"📂 검증용 데이터: {OUTPUT_VALID} ({len(valid_df)}개)")
    print("\n[생성된 데이터 예시]")
    print(train_df.head())

if __name__ == "__main__":
    main()