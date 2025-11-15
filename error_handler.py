import logging
import traceback
from functools import wraps

logger = logging.getLogger(__name__)

def handle_unpacking_errors(func):
    """Декоратор для обработки ошибок распаковки"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                logger.error(f"Ошибка распаковки в {func.__name__}: {e}")
                logger.error(traceback.format_exc())
                # Возвращаем значения по умолчанию вместо падения
                if "expected 3" in str(e):
                    return None, 0.0, None
                elif "expected 2" in str(e):
                    return None, None
            raise e
    return wrapper

def safe_unpack_three(defaults=(None, 0.0, None)):
    """Безопасная распаковка трех значений"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if len(result) == 3:
                    return result
                else:
                    logger.warning(f"Функция {func.__name__} вернула {len(result)} значений вместо 3")
                    return defaults
            except Exception as e:
                logger.error(f"Ошибка в {func.__name__}: {e}")
                return defaults
        return wrapper
    return decorator

def safe_unpack_two(defaults=(None, None)):
    """Безопасная распаковка двух значений"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if len(result) == 2:
                    return result
                else:
                    logger.warning(f"Функция {func.__name__} вернула {len(result)} значений вместо 2")
                    return defaults
            except Exception as e:
                logger.error(f"Ошибка в {func.__name__}: {e}")
                return defaults
        return wrapper
    return decorator