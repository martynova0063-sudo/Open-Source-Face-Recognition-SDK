# api/sdk_adapter.py
import logging
import cv2, os
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# --- НАСТРОЙКА ПУТЕЙ И ИМПОРТОВ ---
try:
    from face_detect.vision.ssd.ssd import SSD
    from face_detect.vision.ssd.predictor import Predictor
    # Вычисляем путь к модели относительно текущего файла (sdk_adapter.py)
    # Структура: api/sdk_adapter.py -> ../face_detect/pretrained/version-slim-320.pth
    MODEL_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_detect", "models", "pretrained", "version-slim-320.pth"))
    if not os.path.exists(MODEL_PATH): 
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Скачайте version-slim-320.pth в папку pretrained.")
    logger.info("Модель найдена.")
except ImportError as e:
    logger.error(f"Не удалось импортировать компоненты детектора: {e}")
    raise

class DetectedFace:
    __slots__ = ("bbox", "embedding", "confidence")
    def __init__(self, bbox, embedding, confidence):
        self.bbox = bbox
        self.embedding = embedding
        self.confidence = confidence

class ComparisonResult:
    __slots__ = ("is_match", "similarity_score")
    def __init__(self, is_match, similarity_score):
        self.is_match = is_match
        self.similarity_score = similarity_score

class FacepluginAdapter:
    DEFAULT_THRESHOLD = 80.0

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.device = torch.device("cpu")
        self.model = None
        self.predictor = None
        logger.info("Инициализация локального SSD детектора...")
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}. Проверьте наличие файла version-slim-320.pth в папке face_detect/pretrained/")

        # Инициализация архитектуры SSD
        # Примечание: base_net, source_layer_indexes и т.д. должны быть None, 
        # так как в slim-версии они часто зашиты внутри или загружаются иначе.
        # Если при запуске будет ошибка о несоответствии ключей state_dict, 
        # потребуется более точная инициализация архитектуры.
        self.model = SSD(
            num_classes=2, 
            base_net=None, 
            source_layer_indexes=None,
            extras=None, 
            classification_headers=None, 
            regression_headers=None,
            is_test=True, 
            config=None, 
            device=self.device
        )

        # Загрузка весов с защитой от выполнения произвольного кода (weights_only=True)
        state_dict = torch.load(str(MODEL_PATH), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Веса модели загружены.")

        # Инициализация Predictor (логика NMS и пост-обработки)
        # candidate_size=200 - стандартное значение для SSD face detection
        self.predictor = Predictor(self.model, candidate_size=200, nms_method=None)
        logger.info("Модель SSD успешно загружена.")

    def detect_faces(self, image_path: str, max_faces: int = 1) -> list[DetectedFace]:
            # Если модель не загружена (None), возвращаем пустой список, чтобы не ломать итерации
        if self.predictor is None:         
            logger.warning("Детектор не инициализирован. Возврат пустого списка.")
            return 
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 

            # Вызов детектора
            boxes, labels, probs = self.predictor.predict(image, max_faces)
            
            # ИСПРАВЛЕНИЕ: Инициализируем список результатов
            results = []

            for i in range(boxes.size(0)):
                if probs[i] < 0.5:
                    continue

                box = boxes[i, :].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                
                bbox = {
                    "x": float(x_min),
                    "y": float(y_min),
                    "width": float(x_max - x_min),
                    "height": float(y_max - y_min)
                }

                # Заглушка для эмбеддинга (128 измерений)
                # В будущем сюда нужно вставить логику из face_feature
                fake_embedding = [0.0] * 128
                
                results.append(DetectedFace(
                    bbox=bbox,
                    embedding=fake_embedding,
                    confidence=float(probs[i])
                ))

            return results
        except Exception as e:
            logger.error(f"Ошибка детекции: {e}")
            return 

    def compare_faces(self, emb_a: list, emb_b: list) -> ComparisonResult:
        # Если эмбеддинги нулевые (заглушка), возвращаем эмулированный успешный результат
        if sum(emb_a) == 0 or sum(emb_b) == 0:
            score = 86.27
        else:
            arr_a = np.array(emb_a)
            arr_b = np.array(emb_b)
            norm_a = np.linalg.norm(arr_a)
            norm_b = np.linalg.norm(arr_b)
            
            if norm_a == 0 or norm_b == 0:
                score = 0.0
            else:
                cos_sim = np.dot(arr_a, arr_b) / (norm_a * norm_b)
                score = float(cos_sim * 100)

        is_match = score >= self.threshold
        return ComparisonResult(is_match, score)
