# compare_video.py
# Сравнение лица с эталонного фото с каждым кадром видео
# Запуск: python compare_video.py --photo test/1.jpg --video test/video.mp4

import cv2
import torch
import numpy as np
import os.path
import time
import argparse

from face_detect.detect_imgs import get_face_boundingbox
from face_landmark.GetLandmark import get_face_landmark
from face_feature.GetFeature import get_face_feature
from face_pose.GetPose import get_face_pose


def GetImageInfo(image, faceMaxCount):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes, scores = get_face_boundingbox(image)
    boxes = boxes[:faceMaxCount]
    scores = scores[:faceMaxCount]
    count = len(boxes)

    bboxes = []
    bscores = []
    landmarks = []
    poses = []
    features = []

    for idx in range(count):
        bboxes.append(boxes[idx].data.numpy())
        bscores.append(scores[idx].data.numpy())
        landmarks.append(get_face_landmark(gray_image, boxes[idx]).data.numpy())
        poses.append(get_face_pose(boxes[idx], landmarks[idx]))
        alignimg, feature = get_face_feature(image, landmarks[idx])
        features.append(feature)

    return count, bboxes, bscores, landmarks, features


def get_similarity(feat1, feat2):
    return (np.sum(feat1 * feat2) + 1) * 50


def main():
    parser = argparse.ArgumentParser(description="Сравнение фото с кадрами видео")
    parser.add_argument("--photo", type=str, required=True, help="Путь к эталонному фото")
    parser.add_argument("--video", type=str, required=True, help="Путь к видео")
    parser.add_argument("--threshold", type=float, default=75.0, help="Порог сходства (по умолчанию 75)")
    parser.add_argument("--step", type=int, default=1, help="Анализировать каждый N-й кадр (по умолчанию каждый)")
    args = parser.parse_args()

    threshold = args.threshold

    # --- 1. Эталонное фото ---
    print(f"Загрузка эталонного фото: {args.photo}")
    ref_img = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"Ошибка: не удалось загрузить фото {args.photo}")
        return

    ref_count, _, _, _, ref_features = GetImageInfo(ref_img, faceMaxCount=1)
    if ref_count == 0:
        print("На эталонном фото лицо не найдено!")
        return

    ref_embedding = ref_features[0]
    print(f"Эталонное лицо извлечено (размер вектора: {ref_embedding.shape})")

    # --- 2. Чтение видео покадрово ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Видео: {total_frames} кадров, {fps:.1f} FPS")
    print(f"Порог сходства: {threshold}")
    print("=" * 50)

    frame_idx = 0
    matches = 0
    checked = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Пропускаем кадры, если задан шаг
        if args.step > 1 and frame_idx % args.step != 0:
            continue

        # Получаем информацию о лицах на кадре
        try:
            count, _, _, _, features = GetImageInfo(frame, faceMaxCount=1)
        except Exception as e:
            print(f"Кадр {frame_idx}: ошибка детекции — {e}")
            continue

        checked += 1

        if count == 0:
            print(f"Кадр {frame_idx:4d} | лицо не найдено")
            continue

        # Сравнение первого (и единственного) лица на кадре с эталоном
        score = get_similarity(ref_embedding, features[0])
        is_same = score >= threshold
        if is_same:
            matches += 1

        status = "✓ Совпадение" if is_same else "✗ Другое лицо"
        time_sec = frame_idx / fps if fps > 0 else 0
        print(f"Кадр {frame_idx:4d} | {time_sec:6.1f}s | Сходство: {score:6.2f}% | {status}")

    cap.release()

    print("=" * 50)
    print(f"Проверено кадров: {checked}")
    print(f"Совпадений: {matches}")
    print(f"Точность: {matches / checked * 100:.1f}%" if checked > 0 else "Нет данных")


if __name__ == "__main__":
    main()
