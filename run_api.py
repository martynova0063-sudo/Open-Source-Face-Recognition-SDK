# run_api.py
import argparse
import logging
import sys
from pathlib import Path

import uvicorn

# Добавляем корень проекта в PYTHONPATH, чтобы импорты из api/ работали корректно
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск API для верификации лиц на базе Faceplugin SDK"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для прослушивания (по умолчанию: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для прослушивания (по умолчанию: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Включить автоперезагрузку при изменении кода (для разработки)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Включить подробный лог (DEBUG)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Запуск Face Recognition SDK API")
    logger.info(f"Host: {args.host}, Port: {args.port}, Reload: {args.reload}")

    try:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="debug" if args.verbose else "info",
        )
    except KeyboardInterrupt:
        logger.info("Остановка сервера по сигналу пользователя")
    except Exception as e:
        logger.critical("Критическая ошибка при запуске: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
