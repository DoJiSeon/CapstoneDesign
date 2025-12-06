# main.py
import shutil
import os
import traceback # [추가] 에러 추적용
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from inference_server import AVSRInferenceHandler

handler = AVSRInferenceHandler()
app = FastAPI()

UPLOAD_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # 모델 로딩
    if handler.modelmodule is None:
        print("💡 첫 요청 감지: 모델 로딩을 시작합니다...")
        try:
            handler.load_model()
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"❌ 모델 로딩 실패:\n{err_msg}")
            return {"status": "error", "inference_result": f"모델 로딩 실패: {str(e)}"}

    file_location = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    print(f"🎬 영상 수신 완료: {file_location}")

    # 추론 실행
    try:
        result_text = handler.inference(file_location)
        status = "success"
    except Exception as e:
        # [핵심] 에러 상세 내용을 터미널에 출력
        err_msg = traceback.format_exc()
        print("="*50)
        print("❌ 추론 도중 에러 발생!")
        print(err_msg)
        print("="*50)
        
        result_text = f"에러 발생: {str(e)}"
        if not str(e): # 에러 메시지가 비어있다면 타입이라도 출력
            result_text = f"에러 발생 (메시지 없음): {type(e).__name__}"
        status = "error"
    
    return {
        "status": status,
        "filename": file.filename, 
        "inference_result": result_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)