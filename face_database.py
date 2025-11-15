import sqlite3
import os
import json
import base64
from datetime import datetime
from contextlib import contextmanager
import logging
from PIL import Image
import io
import sys
import os

def get_base_path():
    """Получение базового пути"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class KaleidoDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = get_base_path()
            self.db_path = os.path.join(base_dir, "data/database.db")
        else:
            self.db_path = db_path
            
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # ... остальной код без изменений
logger = logging.getLogger(__name__)

class KaleidoDatabase:
    def __init__(self, db_path="data/database.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для работы с базой данных"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Инициализация структуры базы данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица людей с дополнительными полями
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_name TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    middle_name TEXT,
                    age INTEGER,
                    position TEXT,
                    department TEXT,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    notes TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Таблица фотографий пользователей с хранением изображений в базе
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    image_data BLOB NOT NULL,  -- Само изображение в формате BLOB
                    image_format TEXT NOT NULL, -- Формат изображения (JPEG, PNG)
                    original_filename TEXT,
                    face_embedding BLOB,        -- Эмбеддинг лица
                    is_primary BOOLEAN DEFAULT 0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE CASCADE
                )
            ''')
            
            # Таблица сессий распознавания
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    recognition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    camera_id TEXT,
                    FOREIGN KEY (person_id) REFERENCES people (id)
                )
            ''')
            
            # Таблица системных настроек
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    description TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Индексы для улучшения производительности
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_people_names ON people(last_name, first_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_person ON person_photos(person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_person_time ON recognition_sessions(person_id, recognition_time)')
            
            # Инициализация настроек по умолчанию
            self._init_default_settings(cursor)
            
            logger.info("KaleidoID Database initialized successfully")

    def _init_default_settings(self, cursor):
        """Инициализация настроек по умолчанию"""
        default_settings = [
            ('recognition_threshold', '0.6', 'Порог распознавания лиц'),
            ('min_detection_confidence', '0.5', 'Минимальная уверенность детекции'),
            ('camera_id', '0', 'ID камеры по умолчанию'),
            ('auto_save_embeddings', '1', 'Автоматически сохранять эмбеддинги'),
            ('max_image_size', '1024', 'Максимальный размер изображения'),
        ]
        
        for key, value, description in default_settings:
            cursor.execute('''
                INSERT OR IGNORE INTO system_settings (key, value, description)
                VALUES (?, ?, ?)
            ''', (key, value, description))

    def get_setting(self, key, default=None):
        """Получение значения настройки"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM system_settings WHERE key = ?', (key,))
            result = cursor.fetchone()
            return result['value'] if result else default

    def set_setting(self, key, value):
        """Установка значения настройки"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO system_settings (key, value, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))

    def safe_get(self, data, key, default=None):
        """Безопасное получение значения из словаря"""
        value = data.get(key, default)
        return value if value is not None else default

    def add_person(self, person_data):
        """Добавление нового человека в базу данных"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO people 
                    (last_name, first_name, middle_name, age, position, 
                     department, phone, email, address, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.safe_get(person_data, 'last_name', '').strip(),
                    self.safe_get(person_data, 'first_name', '').strip(),
                    self.safe_get(person_data, 'middle_name', '').strip(),
                    self.safe_get(person_data, 'age'),
                    self.safe_get(person_data, 'position', '').strip(),
                    self.safe_get(person_data, 'department', '').strip(),
                    self.safe_get(person_data, 'phone', '').strip(),
                    self.safe_get(person_data, 'email', '').strip(),
                    self.safe_get(person_data, 'address', '').strip(),
                    self.safe_get(person_data, 'notes', '').strip()
                ))
                
                person_id = cursor.lastrowid
                logger.info(f"Added person: {self.safe_get(person_data, 'last_name')} {self.safe_get(person_data, 'first_name')} (ID: {person_id})")
                return person_id
                
        except Exception as e:
            logger.error(f"Error adding person: {e}")
            return None

    def update_person(self, person_id, person_data):
        """Обновление данных человека"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE people 
                    SET last_name=?, first_name=?, middle_name=?, age=?, 
                        position=?, department=?, phone=?, email=?, address=?, notes=?,
                        last_updated=CURRENT_TIMESTAMP
                    WHERE id=?
                ''', (
                    self.safe_get(person_data, 'last_name', ''),
                    self.safe_get(person_data, 'first_name', ''),
                    self.safe_get(person_data, 'middle_name', ''),
                    self.safe_get(person_data, 'age'),
                    self.safe_get(person_data, 'position', ''),
                    self.safe_get(person_data, 'department', ''),
                    self.safe_get(person_data, 'phone', ''),
                    self.safe_get(person_data, 'email', ''),
                    self.safe_get(person_data, 'address', ''),
                    self.safe_get(person_data, 'notes', ''),
                    person_id
                ))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Updated person ID: {person_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error updating person {person_id}: {e}")
            return False

    def add_person_photo(self, person_id, image_data, image_format="JPEG", original_filename=None, embedding=None, is_primary=False):
        """Добавление фотографии пользователя в базу данных"""
        try:
            # Если передано изображение как PIL Image, конвертируем в bytes
            if hasattr(image_data, 'save'):
                img_byte_arr = io.BytesIO()
                image_data.save(img_byte_arr, format=image_format)
                image_bytes = img_byte_arr.getvalue()
            else:
                image_bytes = image_data

            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Если это основное фото, снимаем флаг с других фото
                if is_primary:
                    cursor.execute('UPDATE person_photos SET is_primary=0 WHERE person_id=?', (person_id,))
                
                cursor.execute('''
                    INSERT INTO person_photos 
                    (person_id, image_data, image_format, original_filename, face_embedding, is_primary)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person_id, image_bytes, image_format, original_filename, embedding, is_primary))
                
                photo_id = cursor.lastrowid
                logger.info(f"Added photo for person {person_id}, photo ID: {photo_id}")
                return photo_id
                
        except Exception as e:
            logger.error(f"Error adding person photo: {e}")
            return None

    def add_person_photo_from_file(self, person_id, file_path, embedding=None, is_primary=False):
        """Добавление фотографии из файла"""
        try:
            with Image.open(file_path) as img:
                # Конвертируем в RGB если нужно
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Получаем формат файла
                image_format = img.format or 'JPEG'
                original_filename = os.path.basename(file_path)
                
                return self.add_person_photo(
                    person_id, img, image_format, original_filename, embedding, is_primary
                )
        except Exception as e:
            logger.error(f"Error adding photo from file {file_path}: {e}")
            return None

    def get_person_photos(self, person_id):
        """Получение всех фотографий пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, person_id, image_format, original_filename, 
                       face_embedding, is_primary, created_date
                FROM person_photos 
                WHERE person_id=? 
                ORDER BY is_primary DESC, created_date DESC
            ''', (person_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_photo_data(self, photo_id):
        """Получение данных изображения по ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_data, image_format, original_filename
                FROM person_photos 
                WHERE id=?
            ''', (photo_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_photo_as_image(self, photo_id):
        """Получение изображения как PIL Image"""
        photo_data = self.get_photo_data(photo_id)
        if photo_data and photo_data['image_data']:
            return Image.open(io.BytesIO(photo_data['image_data']))
        return None

    def get_primary_photo(self, person_id):
        """Получение основной фотографии пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM person_photos 
                WHERE person_id=? AND is_primary=1
                LIMIT 1
            ''', (person_id,))
            row = cursor.fetchone()
            return row['id'] if row else None

    def set_primary_photo(self, photo_id):
        """Установка фотографии как основной"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Получаем person_id для этой фотографии
                cursor.execute('SELECT person_id FROM person_photos WHERE id=?', (photo_id,))
                result = cursor.fetchone()
                if not result:
                    return False
                
                person_id = result['person_id']
                
                # Снимаем флаг со всех фото пользователя
                cursor.execute('UPDATE person_photos SET is_primary=0 WHERE person_id=?', (person_id,))
                
                # Устанавливаем основное фото
                cursor.execute('UPDATE person_photos SET is_primary=1 WHERE id=?', (photo_id,))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Set photo {photo_id} as primary for person {person_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error setting primary photo: {e}")
            return False

    def delete_photo(self, photo_id):
        """Удаление фотографии"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM person_photos WHERE id=?', (photo_id,))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Deleted photo ID: {photo_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}")
            return False

    def update_photo_embedding(self, photo_id, embedding):
        """Обновление эмбеддинга для фотографии"""
        try:
            embedding_data = None
            if embedding is not None:
                if hasattr(embedding, 'tobytes'):
                    embedding_data = embedding.tobytes()
                else:
                    embedding_data = json.dumps(embedding.tolist()).encode('utf-8') if hasattr(embedding, 'tolist') else None
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE person_photos 
                    SET face_embedding=?
                    WHERE id=?
                ''', (embedding_data, photo_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Updated embedding for photo {photo_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error updating photo embedding: {e}")
            return False

    def get_photo_embedding(self, photo_id):
        """Получение эмбеддинга фотографии"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT face_embedding FROM person_photos WHERE id=?', (photo_id,))
            result = cursor.fetchone()
            if result and result['face_embedding']:
                try:
                    # Пробуем декодировать как numpy array
                    import numpy as np
                    return np.frombuffer(result['face_embedding'], dtype=np.float32)
                except:
                    # Пробуем декодировать как JSON
                    try:
                        return json.loads(result['face_embedding'].decode('utf-8'))
                    except:
                        return None
            return None

    def delete_person(self, person_id):
        """Мягкое удаление человека"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE people SET is_active=0 WHERE id=?', (person_id,))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Deleted person ID: {person_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error deleting person {person_id}: {e}")
            return False

    def get_person(self, person_id):
        """Получение данных конкретного человека"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM people WHERE id=? AND is_active=1', (person_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_people(self, include_inactive=False):
        """Получение всех записей"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if include_inactive:
                cursor.execute('SELECT * FROM people ORDER BY last_name, first_name')
            else:
                cursor.execute('SELECT * FROM people WHERE is_active=1 ORDER BY last_name, first_name')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_people(self, search_term, include_inactive=False):
        """Поиск людей по различным полям"""
        if not search_term:
            return self.get_all_people(include_inactive)
            
        search_pattern = f'%{search_term}%'
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if include_inactive:
                cursor.execute('''
                    SELECT * FROM people 
                    WHERE (last_name LIKE ? OR first_name LIKE ? OR 
                           middle_name LIKE ? OR position LIKE ? OR 
                           department LIKE ? OR phone LIKE ? OR 
                           email LIKE ? OR notes LIKE ?)
                    ORDER BY last_name, first_name
                ''', (search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern))
            else:
                cursor.execute('''
                    SELECT * FROM people 
                    WHERE (last_name LIKE ? OR first_name LIKE ? OR 
                           middle_name LIKE ? OR position LIKE ? OR 
                           department LIKE ? OR phone LIKE ? OR 
                           email LIKE ? OR notes LIKE ?)
                    AND is_active=1
                    ORDER BY last_name, first_name
                ''', (search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern, search_pattern,
                      search_pattern, search_pattern))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def add_recognition_session(self, person_id, confidence, camera_id=None):
        """Добавление записи о распознавании"""
        try:
            conf_value = 0.0
            if confidence is not None:
                try:
                    conf_value = float(confidence)
                except (TypeError, ValueError):
                    conf_value = 0.0
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recognition_sessions (person_id, confidence, camera_id)
                    VALUES (?, ?, ?)
                ''', (person_id, conf_value, camera_id))
                logger.debug(f"Added recognition session for person {person_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding recognition session: {e}")
            return False

    def get_recognition_stats(self, person_id=None, days=30):
        """Получение статистики распознавания"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if person_id:
                cursor.execute('''
                    SELECT COUNT(*) as count, 
                           COALESCE(AVG(confidence), 0) as avg_confidence,
                           MAX(recognition_time) as last_seen
                    FROM recognition_sessions 
                    WHERE person_id=? AND recognition_time >= datetime('now', ?)
                ''', (person_id, f'-{days} days'))
            else:
                cursor.execute('''
                    SELECT COUNT(*) as count, 
                           COALESCE(AVG(confidence), 0) as avg_confidence
                    FROM recognition_sessions
                    WHERE recognition_time >= datetime('now', ?)
                ''', (f'-{days} days',))
                
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['count'] = result.get('count', 0) or 0
                result['avg_confidence'] = float(result.get('avg_confidence', 0.0) or 0.0)
                result['last_seen'] = result.get('last_seen', '')
                return result
            else:
                return {'count': 0, 'avg_confidence': 0.0, 'last_seen': ''}

    def get_database_stats(self):
        """Получение общей статистики базы данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM people WHERE is_active=1')
            total_row = cursor.fetchone()
            total_people = total_row['total'] if total_row and total_row['total'] is not None else 0
            
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) as with_embeddings 
                FROM person_photos 
                WHERE face_embedding IS NOT NULL
            ''')
            embeddings_row = cursor.fetchone()
            with_embeddings = embeddings_row['with_embeddings'] if embeddings_row and embeddings_row['with_embeddings'] is not None else 0
            
            cursor.execute('SELECT COUNT(*) as total_photos FROM person_photos')
            photos_row = cursor.fetchone()
            total_photos = photos_row['total_photos'] if photos_row else 0
            
            # Размер базы данных в MB
            db_size = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            
            cursor.execute('''
                SELECT COUNT(*) as total_sessions,
                       COALESCE(AVG(confidence), 0) as avg_confidence
                FROM recognition_sessions
            ''')
            sessions_row = cursor.fetchone()
            if sessions_row:
                total_sessions = sessions_row['total_sessions'] or 0
                avg_confidence = float(sessions_row['avg_confidence'] or 0.0)
            else:
                total_sessions = 0
                avg_confidence = 0.0
            
            return {
                'total_people': total_people,
                'with_embeddings': with_embeddings,
                'total_photos': total_photos,
                'total_sessions': total_sessions,
                'avg_confidence': avg_confidence,
                'db_size_mb': round(db_size, 2)
            }

    def get_person_with_photos(self, person_id):
        """Получение данных человека вместе с фотографиями"""
        person = self.get_person(person_id)
        if person:
            person['photos'] = self.get_person_photos(person_id)
        return person

    def export_person_data(self, person_id, export_dir="data/exports"):
        """Экспорт данных человека"""
        try:
            os.makedirs(export_dir, exist_ok=True)
            person = self.get_person_with_photos(person_id)
            
            if not person:
                return False
            
            # Создаем папку для экспорта
            safe_name = f"{person['last_name']}_{person['first_name']}_{person_id}".replace(' ', '_')
            person_dir = os.path.join(export_dir, safe_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Сохраняем информацию о человеке
            info_file = os.path.join(person_dir, "person_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                # Убираем photos из JSON
                export_data = person.copy()
                if 'photos' in export_data:
                    del export_data['photos']
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Сохраняем фотографии
            photos_dir = os.path.join(person_dir, "photos")
            os.makedirs(photos_dir, exist_ok=True)
            
            for photo_info in person.get('photos', []):
                photo_data = self.get_photo_data(photo_info['id'])
                if photo_data:
                    filename = f"photo_{photo_info['id']}.{photo_data['image_format'].lower()}"
                    filepath = os.path.join(photos_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(photo_data['image_data'])
            
            logger.info(f"Exported person {person_id} to {person_dir}")
            return person_dir
            
        except Exception as e:
            logger.error(f"Error exporting person {person_id}: {e}")
            return False

    def cleanup_old_sessions(self, days=30):
        """Очистка старых сессий распознавания"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM recognition_sessions 
                    WHERE recognition_time < datetime('now', ?)
                ''', (f'-{days} days',))
                
                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} old recognition sessions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0

    def backup_database(self, backup_dir="data/backups"):
        """Создание резервной копии базы данных"""
        try:
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"kaleidodb_backup_{timestamp}.db")
            
            with self.get_connection() as conn:
                # Используем встроенную функцию SQLite для бэкапа
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return None