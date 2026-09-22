# api/models.py
from typing import List, Optional
from pydantic import BaseModel, Field


class FrameMatchResult(BaseModel):
    """Результат анализа одного кадра видео."""
    frame_index: int = Field(..., description="Индекс кадра в последовательности (0-based)")
    timestamp_sec: float = Field(..., description="Время кадра в секундах от начала видео")
    similarity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Оценка сходства (0–100). Чем выше, тем больше похоже на эталон."
    )
    is_match: bool = Field(..., description="Прошёл ли кадр порог верификации")
    face_detected: bool = Field(..., description="Обнаружено ли лицо на кадре")


class VerificationResponse(BaseModel):
    """Итоговый ответ эндпоинта /verify."""
    verified: bool = Field(
        ...,
        description=(
            "Агрегированный вердикт. Сейчас считается True, если хотя бы один кадр "
            "дал совпадение. Логику можно ужесточить под требования системы."
        )
    )
    best_similarity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Максимальное значение similarity среди всех кадров"
    )
    threshold: float = Field(
        ...,
        ge=0,
        le=100,
        description="Порог, который использовался для принятия решения"
    )
    total_frames_checked: int = Field(..., description="Сколько кадров было проанализировано")
    frames_with_faces: int = Field(..., description="На скольких кадрах было найдено лицо")
    matched_frames: int = Field(..., description="Сколько кадров дали совпадение (is_match=True)")
    per_frame_results: List[FrameMatchResult] = Field(
        ...,
        description="Детальная разбивка по каждому обработанному кадру"
    )
    reference_face_detected: bool = Field(
        ...,
        description="Было ли найдено лицо на эталонном фото"
    )
    error: Optional[str] = Field(
        None,
        description="Сообщение об ошибке (если произошла). Если None — ошибок нет."
    )
