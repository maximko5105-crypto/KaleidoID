#!/usr/bin/env python3
"""
Главный модуль системы распознавания лиц KaleidoID
"""

import tkinter as tk
import logging
import os
import sys

def get_base_path():
    """Получение базового пути для работы в EXE режиме"""
    if getattr(sys, 'frozen', False):
        # Если приложение запущено как EXE
        return os.path.dirname(sys.executable)
    else:
        # Если приложение запущено как скрипт
        return os.path.dirname(os.path.abspath(__file__))

# Добавляем путь к src в PYTHONPATH
BASE_DIR = get_base_path()
src_path = os.path.join(BASE_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def setup_logging():
    """Настройка системы логирования"""
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'kaleidoid.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Проверка зависимостей"""
    try:
        import cv2
        import mediapipe
        import PIL
        import numpy
        print("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False

def create_necessary_dirs():
    """Создание необходимых директорий"""
    directories = [
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "data/exports"),
        os.path.join(BASE_DIR, "data/backups"),
        os.path.join(BASE_DIR, "logs")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Главная функция приложения"""
    print("🔮 Запуск KaleidoID - Advanced Face Recognition System")
    print("🚀 Версия с landmarks и захватом фотографий")
    
    # Создание директорий
    create_necessary_dirs()
    
    # Проверка зависимостей
    if not check_dependencies():
        input("Нажмите Enter для выхода...")
        return
    
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        from database.face_database import KaleidoDatabase
        from recognition.face_recognizer import KaleidoRecognizer
        from gui.main_window import KaleidoIDGUI
        
        # Инициализация компонентов
        logger.info("Инициализация базы данных...")
        db_path = os.path.join(BASE_DIR, "data/database.db")
        database = KaleidoDatabase(db_path=db_path)
        
        logger.info("Инициализация распознавателя лиц...")
        recognizer = KaleidoRecognizer(
            min_detection_confidence=float(database.get_setting('min_detection_confidence', 0.5))
        )
        
        # Загрузка эмбеддингов из базы данных
        logger.info("Загрузка эмбеддингов из базы данных...")
        loaded_count = recognizer.load_embeddings_from_database(database)
        logger.info(f"Загружено {loaded_count} эмбеддингов")
        
        # Создание графического интерфейса
        logger.info("Создание графического интерфейса...")
        root = tk.Tk()
        app = KaleidoIDGUI(root, database, recognizer)
        
        # Настройка обработчика закрытия
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Центрирование окна
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
        root.geometry(f"+{x}+{y}")
        
        logger.info("Запуск главного цикла приложения...")
        
        # Запускаем интерфейс
        app.run()
        
        logger.info("Приложение KaleidoID завершено")
        
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске: {e}")
        print(f"❌ Ошибка запуска KaleidoID: {e}")
        import traceback
        traceback.print_exc()
        
        input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()