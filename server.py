from flask import Flask, request, jsonify, render_template
import subprocess
import os

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    video = request.files["video"]
    audio = request.files["audio"]

    video_path = os.path.join(UPLOAD_DIR, "video.mp4")
    audio_path = os.path.join(UPLOAD_DIR, "audio.wav")

    video.save(video_path)
    audio.save(audio_path)

    # inference 실행
    result = subprocess.check_output([
        "python", "inference_avsr.py",
        "--checkpoint", "AVSR_Whisper-M_AVH-L_Llama3.1-8B_lrs3vox_Adown4_Vdown2_seed42.pth",
        "--modality", "audiovisual",
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--use-uadf"
    ])

    return jsonify({"text": result.decode().strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
