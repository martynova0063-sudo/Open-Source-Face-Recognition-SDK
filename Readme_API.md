# Основной запуск FastAPI 

```bash
python run_api.py --host 0.0.0.0 --port 8000
```

Реализован всего один метод

# Основной рабочий файл video_compare.py


## Структура файлов

| Путь | Описание |
|---|---|
| `photos\YYYY\MM\DD\<employeeUuid>\<eventUuid>_photo.jpg` | Эталонное фото |
| `videos\YYYY\MM\DD\<employeeUuid>\<eventUuid>_video.mp4` | Видео осмотра |

## Расшифровка

- **Папка `<employeeUuid>`** — идентификатор работника.
- **Имя файла `<eventUuid>`** — идентификатор события верификации; связывает фото и видео одной пары.
- Фото и видео с одинаковым `eventUuid` — это пара «эталон + видео» для одного осмотра.


# Альтернативный запуск

python compare_video.py --photo photos/2026/08/28/1e60a4d4-2aaf-11f0-a2c4-6a4b6c56230e/3ef08bb5-a298-11f1-9cae-6a4b6c56230e_photo.jpg --video videos/2026/08/28/1e60a4d4-2aaf-11f0-a2c4-6a4b6c56230e/3ef08bb5-a298-11f1-9cae-6a4b6c56230e_video.mp4

# ID расшифровка результата

outputs.md



