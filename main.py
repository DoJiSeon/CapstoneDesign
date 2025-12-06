# main.py 수정본 (2번 방식 적용)

import shutil
import os
import traceback
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from inference_server import AVSRInferenceHandler

handler = AVSRInferenceHandler()
MODEL_STATUS = "not_started" # 상태 관리 변수

async def load_model_background():
    """백그라운드에서 모델을 로딩하는 함수"""
    global MODEL_STATUS
    print("⏳ [Background] 모델 로딩 시작...")
    MODEL_STATUS = "loading"
    try:
        # 동기 함수인 load_model을 비동기 환경에서 실행 (스레드 풀 사용)
        await asyncio.to_thread(handler.load_model)
        MODEL_STATUS = "ready"
        print("✅ [Background] 모델 로딩 완료! 추론 가능.")
    except Exception as e:
        MODEL_STATUS = "error"
        print(f"❌ [Background Error] 모델 로딩 실패: {e}")
        print(traceback.format_exc())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 백그라운드 작업으로 로딩 시작 (기다리지 않음!)
    asyncio.create_task(load_model_background())
    yield
    # 서버 종료 시
    print("👋 서버 종료")

app = FastAPI(lifespan=lifespan)

# (이하 설정 코드는 동일)
UPLOAD_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 웹페이지에 현재 상태를 같이 내려줄 수도 있음 (HTML에서 처리 필요)
    return templates.TemplateResponse("index.html", {"request": request, "status": MODEL_STATUS})

# [추가] 현재 모델 상태를 확인하는 API (프론트엔드에서 폴링용)
@app.get("/status")
async def check_status():
    return {"status": MODEL_STATUS}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    global MODEL_STATUS
    
    # 1. 모델이 아직 로딩 중인 경우
    if MODEL_STATUS == "loading" or handler.modelmodule is None:
        return {
            "status": "loading",
            "inference_result": "⏳ 모델을 서버 메모리에 올리는 중입니다... (약 20초 소요)\n잠시 후 다시 시도해 주세요."
        }
    
    # 2. 모델 로딩에 실패한 경우
    if MODEL_STATUS == "error":
        return {
            "status": "error",
            "inference_result": "❌ 서버 내부 오류로 모델 로딩에 실패했습니다. 관리자에게 문의하세요."
        }

    # 3. 정상 추론 진행
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    print(f"🎬 영상 수신 완료: {file_location}")

    try:
        # 비동기로 추론 실행 (FastAPI 서버 멈춤 방지)
        result_text = await asyncio.to_thread(handler.inference, file_location)
        status = "success"
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ 추론 에러: {err_msg}")
        result_text = f"에러 발생: {str(e)}"
        status = "error"
    
    return {
        "status": status,
        "filename": file.filename, 
        "inference_result": result_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)