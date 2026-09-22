# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="Face Recognition SDK API",
    description=(
        "REST API для верификации лиц на базе Open-Source Face Recognition SDK от Faceplugin.\n\n"
        "Принимает эталонное фото сотрудника и видео события, "
        "извлекает кадры, сравнивает лица и возвращает результат верификации."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — если фронтенд будет на другом домене
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # В продакшене ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роуты
app.include_router(router)


@app.get("/health", tags=["Служебные"], summary="Проверка работоспособности сервиса.")
async def health_check():
    return {"status": "ok", "service": "face-recognition-sdk-api"}
