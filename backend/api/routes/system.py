"""System-level endpoints (GPU detection, etc.)."""
from fastapi import APIRouter

from backend.dict_gpu.runtime import detect_gpu

router = APIRouter()


@router.get("/gpu", summary="GPU detection and VRAM probe")
def gpu_status() -> dict:
    """Return CUDA device info or {available: false} if no CUDA."""
    s = detect_gpu()
    return {
        "available": s.available,
        "name": s.name,
        "vram_total_gb": s.vram_total_gb,
        "vram_free_gb": s.vram_free_gb,
    }
