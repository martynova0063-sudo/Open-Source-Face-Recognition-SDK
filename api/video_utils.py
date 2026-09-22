# api/video_utils.py
from __future__ import annotations

import logging
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    max_frames: int = 30,
    min_interval_sec: float = 0.5,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Итеративно извлекает кадры из видеофайла.

    Берёт не каждый кадр, а с шагом min_interval_sec —
    чтобы не тратить время на дубликаты и не перегружать SDK.

    Параметры:
        video_path        — путь к видеофайлу (MP4, AVI, MOV).
        max_frames        — максимум кадров для извлечения (по умолчанию 30).
        min_interval_sec  — минимальный интервал между кадрами в секундах.
                            0.5 = брать не чаще одного кадра в полсекунды.

    Возвращает (yield):
        frame_index  — порядковый номер извлечённого кадра (0-based).
        timestamp_sec — время кадра от начала видео (секунды).
        frame        — кадр как np.ndarray (BGR, формат OpenCV).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Не удалось открыть видео: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN-проверка
        fps = 25.0
        logger.warning("FPS не определён, используется значение по умолчанию: %.1f", fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(fps * min_interval_sec))

    logger.info(
        "Извлечение кадров: fps=%.1f, total_frames=%d, frame_skip=%d, max_frames=%d",
        fps,
        total_frames,
        frame_skip,
        max_frames,
    )

    frame_index = 0       # Порядковый номер отданного кадра
    current_frame = 0     # Текущий кадр в потоке видео
    yielded = 0

    try:
        while yielded < max_frames:
            # Перематываем к нужному кадру
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret:
                # Конец видео или сбой чтения
                break

            timestamp_sec = current_frame / fps

            yield frame_index, timestamp_sec, frame

            frame_index += 1
            yielded += 1
            current_frame += frame_skip

    except Exception as e:
        logger.error("Ошибка при извлечении кадров: %s", e)
    finally:
        cap.release()
        logger.info("Извлечение завершено: отдано %d кадров", yielded)
