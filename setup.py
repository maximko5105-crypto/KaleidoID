#!/usr/bin/env python3
"""
Скрипт установки KaleidoID
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    print(f"✅ Python {platform.python_version()}")
    return True

def install_requirements():
    """Установка зависимостей"""
    print("📦 Установка зависимостей...")
    
    requirements = [
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0", 
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "pyinstaller>=5.0.0"
    ]
    
    for package in requirements:
        try:
            package_name = package.split('>=')[0] if '>=' in package else package
            __import__(package_name)
            print(f"✅ {package_name} уже установлен")
        except ImportError:
            print(f"📥 Установка {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} установлен")
            except subprocess.CalledProcessError:
                print(f"❌ Ошибка установки {package}")
                return False
    return True

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        "data",
        "data/exports", 
        "data/backups",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Создана директория: {directory}")
        else:
            print(f"✅ Директория существует: {directory}")

def main():
    """Основная функция установки"""
    print("🔮 Установка KaleidoID")
    print("=" * 50)
    
    # Проверка Python
    if not check_python_version():
        return
    
    # Создание директорий
    create_directories()
    
    # Установка зависимостей
    if not install_requirements():
        print("❌ Ошибка установки зависимостей")
        return
    
    print("\n🎉 Установка завершена!")
    print("\n🚀 Запуск приложения:")
    print("   python run.py")
    print("\n🔧 Сборка EXE:")
    print("   python build_exe.py")

if __name__ == "__main__":
    main()