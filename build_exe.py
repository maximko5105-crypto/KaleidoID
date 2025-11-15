#!/usr/bin/env python3
"""
Скрипт сборки KaleidoID в EXE
"""

import os
import sys
import shutil
import subprocess
import platform

def cleanup_build_dirs():
    """Очистка временных директорий сборки"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"🧹 Очистка {dir_name}...")
            shutil.rmtree(dir_name)
    
    # Очищаем pycache в поддиректориях
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                print(f"🧹 Очистка {pycache_path}...")
                shutil.rmtree(pycache_path)

def build_with_pyinstaller():
    """Сборка с помощью PyInstaller"""
    print("🔨 Запуск PyInstaller...")
    
    # Сначала создаем spec файл если его нет
    if not os.path.exists('kaleido_id.spec'):
        print("📝 Создание spec файла...")
        subprocess.run([
            'pyinstaller', 
            '--name=KaleidoID',
            '--onefile',
            '--windowed',
            '--add-data=src;src',
            '--add-data=data;data',
            '--hidden-import=tkinter',
            '--hidden-import=PIL',
            '--hidden-import=PIL._tkinter_finder',
            '--hidden-import=cv2',
            '--hidden-import=mediapipe',
            '--hidden-import=numpy',
            'run.py'
        ], check=True)
    else:
        # Используем существующий spec файл
        subprocess.run(['pyinstaller', 'kaleido_id.spec', '--noconfirm'], check=True)
    
    print("✅ Сборка завершена успешно!")
    return True

def main():
    """Основная функция сборки"""
    print("🔮 Сборка KaleidoID в EXE")
    print("=" * 50)
    
    # Очистка
    cleanup_build_dirs()
    
    # Сборка
    if build_with_pyinstaller():
        print("\n🎉 Сборка завершена!")
        print(f"📁 EXE файл: dist/KaleidoID.exe")
    else:
        print("\n❌ Сборка не удалась")

if __name__ == "__main__":
    main()