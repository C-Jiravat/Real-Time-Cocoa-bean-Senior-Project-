from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cocoa_platform.auth.service import issue_token, validate_token, verify_password
from cocoa_platform.benchmark import BenchmarkInputError, Sample, evaluate_samples, samples_from_zip
from cocoa_platform.config.settings import get_settings
from cocoa_platform.inference import InferencePipeline
from cocoa_platform.inference.pipeline import InferenceUnavailable


app = FastAPI(title="Cocoa Bean AI MVP", version="1.0.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=1)


@lru_cache
def get_pipeline() -> InferencePipeline:
    return InferencePipeline()


def require_admin(cocoa_session: str | None = Cookie(default=None)) -> str:
    current_settings = get_settings()
    if not current_settings.auth_secret:
        raise HTTPException(503, "ยังไม่ได้ตั้งค่า COCOA_AUTH_SECRET")
    email = validate_token(cocoa_session or "", current_settings.auth_secret)
    if email != current_settings.admin_email:
        raise HTTPException(401, "กรุณาเข้าสู่ระบบ")
    return email


def _http_error(error: Exception) -> HTTPException:
    return HTTPException(400, str(error))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "pipeline": get_pipeline().health()}


@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response) -> dict:
    current_settings = get_settings()
    if not current_settings.auth_secret or not current_settings.admin_password_hash:
        raise HTTPException(503, "ยังไม่ได้ตั้งค่า COCOA_AUTH_SECRET และ COCOA_ADMIN_PASSWORD_HASH")
    if payload.email != current_settings.admin_email or not verify_password(payload.password, current_settings.admin_password_hash):
        raise HTTPException(401, "อีเมลหรือรหัสผ่านไม่ถูกต้อง")
    response.set_cookie("cocoa_session", issue_token(payload.email, current_settings.auth_secret), httponly=True, samesite="lax", max_age=8 * 3600)
    return {"email": payload.email}


@app.post("/api/auth/logout")
def logout(response: Response) -> dict:
    response.delete_cookie("cocoa_session")
    return {"ok": True}


@app.get("/api/auth/me")
def me(email: str = Depends(require_admin)) -> dict:
    return {"email": email}


@app.post("/api/analysis")
async def analysis(
    file: UploadFile = File(...),
    confidence: float = Form(...),
    iou: float = Form(...),
    device: Literal["auto", "cpu", "gpu"] = Form("auto"),
    _: str = Depends(require_admin),
) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "รองรับเฉพาะ JPEG, PNG และ WebP")
    try:
        return get_pipeline().analyze(await file.read(), confidence, iou, device)
    except (ValueError, InferenceUnavailable) as error:
        raise _http_error(error) from error


@app.post("/api/benchmark/single")
async def benchmark_single(
    image: UploadFile = File(...),
    target: Literal["color", "defect", "both"] = Form(...),
    label: UploadFile | None = File(None),
    color_label: UploadFile | None = File(None),
    defect_label: UploadFile | None = File(None),
    confidence: float = Form(...),
    iou: float = Form(...),
    device: Literal["auto", "cpu", "gpu"] = Form("auto"),
    _: str = Depends(require_admin),
) -> dict:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "ต้องเลือกภาพ PNG/JPEG/WebP")
    if target == "both":
        if not color_label or not defect_label or not color_label.filename.lower().endswith(".txt") or not defect_label.filename.lower().endswith(".txt"):
            raise HTTPException(400, "โหมดทั้งสีและข้อบกพร่อง ต้องเลือก color.txt และ defect.txt")
    elif not label or not label.filename.lower().endswith(".txt"):
        raise HTTPException(400, "ต้องเลือก label .txt")
    try:
        sample = Sample(image.filename, await image.read(), await color_label.read() if target == "both" else (await label.read() if target == "color" else None), await defect_label.read() if target == "both" else (await label.read() if target == "defect" else None))
        return {"validation_errors": [], **evaluate_samples(get_pipeline(), [sample], confidence, iou, device, target)}
    except (ValueError, InferenceUnavailable, BenchmarkInputError) as error:
        raise _http_error(error) from error


@app.post("/api/benchmark/zip")
async def benchmark_zip(
    archive: UploadFile = File(...),
    target: Literal["color", "defect", "both"] = Form(...),
    confidence: float = Form(...),
    iou: float = Form(...),
    device: Literal["auto", "cpu", "gpu"] = Form("auto"),
    _: str = Depends(require_admin),
) -> dict:
    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(400, "ต้องเลือกไฟล์ .zip")
    try:
        samples, validation_errors = samples_from_zip(await archive.read(), target)
        return {"validation_errors": validation_errors, **evaluate_samples(get_pipeline(), samples, confidence, iou, device, target)}
    except (ValueError, InferenceUnavailable, BenchmarkInputError) as error:
        raise _http_error(error) from error
