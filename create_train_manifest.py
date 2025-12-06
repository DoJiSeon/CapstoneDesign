import os
import csv

def clean_alignment(path):
    words = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, _, word = parts
            if word == "sil":
                continue
            words.append(word)
    return " ".join(words)


def create_manifest(root, save_path):
    align_dir = os.path.join(root, "align/s1")
    video_dir = os.path.join(root, "video/s1")
    audio_dir = os.path.join(root, "audio/s1")

    items = []

    bases = set([f.split(".")[0] for f in os.listdir(align_dir)])

    for base in bases:
        align = os.path.join(align_dir, base + ".align")
        video = os.path.join(video_dir, base + ".mpg")
        audio = os.path.join(audio_dir, base + ".wav")

        if not (os.path.exists(align) and os.path.exists(video) and os.path.exists(audio)):
            continue

        # Alignment → 텍스트로 변환
        text = clean_alignment(align)

        if len(text.strip()) == 0:
            continue

        items.append([video, audio, text])

    print("총 샘플:", len(items))

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "audio_path", "text"])
        writer.writerows(items)

    print("Manifest 생성 완료:", save_path)


if __name__ == "__main__":
    create_manifest(
        root="dataset",
        save_path="train_manifest.csv"
    )
