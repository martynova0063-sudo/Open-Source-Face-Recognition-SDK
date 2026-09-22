# api/video_compare.py
import cv2
import numpy as np
import csv
import os
import tempfile
import logging
from pathlib import Path

from face_detect.detect_imgs import get_face_boundingbox
from face_landmark.GetLandmark import get_face_landmark
from face_feature.GetFeature import get_face_feature
from face_pose.GetPose import get_face_pose

logger = logging.getLogger(__name__)


def _get_image_info(image, face_max_count=1):
    """Получение эмбеддинга лица из кадра (numpy array)."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes, scores = get_face_boundingbox(image)
    boxes = boxes[:face_max_count]
    scores = scores[:face_max_count]
    count = len(boxes)

    features = []
    for idx in range(count):
        landmarks = get_face_landmark(gray_image, boxes[idx]).data.numpy()
        alignimg, feature = get_face_feature(image, landmarks)
        features.append(feature)

    return count, features


def _get_similarity(feat1, feat2):
    """Косинусное сходство в диапазоне 0–100."""
    return float((np.sum(feat1 * feat2) + 1) * 50)


def compare_photo_with_video(
    photo_path: str,
    video_path: str,
    output_csv: str,
    threshold: float = 75.0,
    step: int = 1,
) -> dict:
    """
    Сравнивает эталонное фото с каждым N-м кадром видео.
    Сохраняет результаты в CSV. Возвращает сводку.
    """
    # --- 1. Эталонное фото ---
    ref_img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        raise ValueError(f"Не удалось загрузить фото: {photo_path}")

    ref_count, ref_features = _get_image_info(ref_img, face_max_count=1)
    if ref_count == 0:
        raise ValueError("На эталонном фото лицо не найдено")

    ref_embedding = ref_features[0]

    # --- 2. Видео ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- 3. CSV ---
    csv_dir = os.path.dirname(output_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    matches = 0
    checked = 0
    no_face = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "time_sec", "similarity_pct", "is_match", "threshold", "status"
        ])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if step > 1 and frame_idx % step != 0:
                continue

            time_sec = round(frame_idx / fps, 3)

            try:
                count, features = _get_image_info(frame, face_max_count=1)
            except Exception as e:
                logger.warning(f"Кадр {frame_idx}: ошибка детекции — {e}")
                writer.writerow([frame_idx, time_sec, "", "", threshold, "error"])
                continue

            checked += 1

            if count == 0:
                no_face += 1
                writer.writerow([frame_idx, time_sec, "", "", threshold, "no_face"])
                continue

            score = _get_similarity(ref_embedding, features[0])
            is_same = score >= threshold
            if is_same:
                matches += 1

            status = "match" if is_same else "mismatch"
            writer.writerow([
                frame_idx, time_sec, f"{score:.2f}", is_same, threshold, status
            ])

    cap.release()

    return {
        "total_frames": total_frames,
        "frames_checked": checked,
        "frames_no_face": no_face,
        "matches": matches,
        "accuracy_pct": round(matches / checked * 100, 1) if checked > 0 else 0.0,
        "threshold": threshold,
        "step": step,
        "csv_path": output_csv,
    }
