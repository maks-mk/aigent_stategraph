import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Импорт Rich для красивого вывода
from rich.logging import RichHandler

class IgnoreSchemaWarnings(logging.Filter):
    """Фильтр для подавления специфических предупреждений."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Key 'additionalProperties' is not supported" not in msg and \
               "Key '$schema' is not supported" not in msg

def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: str = "%(message)s"
) -> logging.Logger:
    
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    if log_file is None:
        log_file = os.getenv("LOG_FILE", "ai_agent.log")

    handlers: List[logging.Handler] = []

    # --- H1: КРАСИВЫЙ КОНСОЛЬНЫЙ ВЫВОД (RICH) ---
    console_handler = RichHandler(
        rich_tracebacks=False,  # <--- ИЗМЕНЕНО: False убирает "мусор" с кодом
        markup=True,
        show_path=False,
        show_time=True,
        omit_repeated_times=False
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console_handler)

    # --- H2: ФАЙЛОВЫЙ ВЫВОД (ДЛЯ ОТЛАДКИ) ---
    # В файл мы всё равно пишем полную ошибку на случай, если нужно разобраться
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG) 
            file_handler.setFormatter(file_fmt)
            handlers.append(file_handler)
        except Exception as e:
            sys.stderr.write(f"⚠️ Не удалось создать лог-файл: {e}\n")

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Фильтры и подавление шума (оставляем как было)
    schema_filter = IgnoreSchemaWarnings()
    for h in handlers:
        h.addFilter(schema_filter)

    noisy_modules = [
        "langchain_mcp_adapters", "mcp", "jsonschema", "langchain_google_genai",
        "httpcore", "httpx", "openai", "chromadb", "hnswlib", 
        "google.ai.generativelanguage", "urllib3", "multipart",
        "sentence_transformers", "filelock", "grpc", "grpc._cython" # <-- Добавил grpc
    ]
    
    lib_level = logging.ERROR if level > logging.DEBUG else logging.WARNING
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(lib_level)

    return logging.getLogger("AgentCore")