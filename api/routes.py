# api/routes.py
from fastapi import APIRouter, HTTPException
from typing import Optional
from .sdk_adapter import FacepluginAdapter
from pydantic import BaseModel
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import tempfile
from .video_compare import compare_photo_with_video
import logging

# Создаем экземпляр роутера
router = APIRouter(prefix="/api", tags=["Face Recognition"])

@router.post("/detect", include_in_schema=False)
async def detect_face(file: UploadFile = File(...)):
    import tempfile
    import os
    
    # Сохраняем загруженный файл во временную папку
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    
    try:
        adapter = FacepluginAdapter()
        results = adapter.detect_faces(temp_path)
        
        response_data = []
        for face in results:
            response_data.append({
                "bbox": face.bbox,
                "confidence": face.confidence,
                "embedding_length": len(face.embedding) # Пока заглушка 128
            })
        return {"faces_found": len(response_data), "data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Модель данных для запроса (ожидаем массивы чисел)
class CompareRequest(BaseModel):
    embedding_a: List[float]
    embedding_b: List[float]

@router.post("/compare", include_in_schema=False)
async def compare_faces(body: CompareRequest):
    """
    Реальное сравнение лиц по эмбеддингам.
    """
    try:
        adapter = FacepluginAdapter()
        result = adapter.compare_faces(body.embedding_a, body.embedding_b)
        
        return {
            "is_match": result.is_match,
            "similarity_score": result.similarity_score,
            "threshold": adapter.threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Video Compare with CSV export ---
@router.post("/video-compare", summary="Сравнивает лицо с эталонного фото с каждым N-м кадром видео. Возвращает JSON и сохраняет результаты в CSV.")
async def video_compare(
    photo: UploadFile = File(..., description="Эталонное фото"),
    video: UploadFile = File(..., description="Видеофайл"),
    threshold: float = Form(75.0, description="Порог сходства"),
    step: int = Form(1, description="Анализировать каждый N-й кадр"),
):

    # Сохраняем загруженные файлы во временную папку
    with tempfile.TemporaryDirectory() as tmpdir:
        photo_path = os.path.join(tmpdir, "reference.jpg")
        video_path = os.path.join(tmpdir, "input_video.mp4")
        csv_path = os.path.join(tmpdir, "results.csv")

        with open(photo_path, "wb") as f:
            f.write(await photo.read())
        with open(video_path, "wb") as f:
            f.write(await video.read())

        try:
            result = compare_photo_with_video(
                photo_path=photo_path,
                video_path=video_path,
                output_csv=csv_path,
                threshold=threshold,
                step=step,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Ошибка сравнения: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        # Читаем CSV для возврата в ответе
        import csv as csv_module
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                rows.append(row)

        # Копируем CSV в папку outputs (чтобы файл остался после очистки tmpdir)
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        final_csv = os.path.join(output_dir, f"compare_{os.path.basename(photo.filename)}_{result['matches']}.csv")
        import shutil
        shutil.copy2(csv_path, final_csv)

    result["csv_filename"] = os.path.basename(final_csv)
    result["csv_rows"] = rows[:50]  # первые 50 строк для превью

    return result    