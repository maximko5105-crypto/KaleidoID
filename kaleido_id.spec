# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys
import os

# Добавляем путь к исходным файлам
sys.path.append(os.path.join(os.path.dirname(__name__), 'src'))

a = Analysis(
    ['run.py'],
    pathex=[os.path.dirname(__name__)],
    binaries=[],
    datas=[
        ('data', 'data'),
        ('logs', 'logs'),
        ('src', 'src')
    ],
    hiddenimports=[
        'PIL',
        'PIL._tkinter_finder',
        'cv2',
        'numpy',
        'mediapipe',
        'tkinter',
        'sqlite3'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 
        'scipy',
        'pytest',
        'unittest',
        'email',
        'http',
        'xml',
        'pydoc'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Добавляем дополнительные файлы
added_files = []

# Добавляем все Python файлы из src
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, '.')
            added_files.append((full_path, relative_path))

a.datas += added_files

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='KaleidoID',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Измените на True для отладки
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='kaleido_icon.ico'  # Добавьте иконку если есть
)