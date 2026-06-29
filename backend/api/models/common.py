"""Common Pydantic models shared across routes."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple


class FileOpenRequest(BaseModel):
    path: str


class FileOpenResponse(BaseModel):
    success: bool
    format_type: str  # 'Oxford' or 'EDAX'
    grid_shape: List[int]  # [n_rows, n_cols]
    pattern_count: int
    pattern_shape: List[int]  # [height, width]
    has_patterns: bool
    has_raw_patterns: bool
    has_eds: bool
    has_electron_images: bool
    eds_elements: List[str]
    electron_images: List[str]
    file_path: str


class PatternResponse(BaseModel):
    image: str  # Base64 PNG
    row: int
    col: int
    index: int


class EDSSpectrumResponse(BaseModel):
    elements: List[str]
    counts: Dict[str, float]
    weight_pct: Dict[str, float]
    atomic_pct: Dict[str, float]


class ElementMapResponse(BaseModel):
    image: str  # Base64 PNG
    element: str
    min_val: float
    max_val: float


class ImageResponse(BaseModel):
    image: str  # Base64 PNG
    width: int
    height: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # 'running', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
