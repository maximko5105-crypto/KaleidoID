import cv2
import mediapipe as mp
import numpy as np
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import os
from PIL import Image
import io

logger = logging.getLogger(__name__)

class KaleidoRecognizer:
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Инициализация MediaPipe для распознавания лиц в KaleidoID
        """
        self.min_detection_confidence = min_detection_confidence
        
        # Инициализация MediaPipe компонентов
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Детектор лиц
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )
        
        # Face Mesh для извлечения признаков и landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Хранилище известных эмбеддингов
        self.known_embeddings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_ids: List[int] = []
        self.known_photo_ids: List[int] = []  # ID фотографий в базе данных
        
        # Настройки распознавания
        self.recognition_threshold: float = 0.6
        self.embedding_size: int = 468 * 3  # 468 landmarks × 3 coordinates
        
        # Кэш для ускорения работы
        self._embedding_cache: Dict[int, np.ndarray] = {}
        
        # Настройки отображения landmarks
        self.show_landmarks: bool = True
        self.landmark_style = {
            'color': (0, 255, 0),  # Зеленый цвет
            'thickness': 1,
            'radius': 1
        }
        
        logger.info("KaleidoID Face Recognizer initialized")

    def safe_float(self, value, default=0.0):
        """Безопасное преобразование в float"""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга лица из изображения
        """
        try:
            # Конвертируем в RGB для MediaPipe
            if len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Обрабатываем изображение для извлечения landmarks
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Берем первое найденное лицо
                face_landmarks = results.multi_face_landmarks[0]
                
                # Извлекаем координаты landmarks
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Преобразуем в numpy array
                landmarks_array = np.array(landmarks, dtype=np.float32)
                
                # Нормализуем эмбеддинг
                if len(landmarks_array) == self.embedding_size:
                    embedding = self._normalize_embedding(landmarks_array)
                    logger.debug("Face embedding extracted successfully")
                    return embedding
            
            logger.warning("No face landmarks detected in image")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def extract_embedding_with_landmarks(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        Извлечение эмбеддинга и landmarks лица из изображения
        
        Returns:
            Tuple[embedding, landmarks] где landmarks - список точек (x, y)
        """
        try:
            # Конвертируем в RGB для MediaPipe
            if len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Обрабатываем изображение для извлечения landmarks
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Берем первое найденное лицо
                face_landmarks = results.multi_face_landmarks[0]
                
                # Извлекаем координаты landmarks для эмбеддинга
                embedding_landmarks = []
                for landmark in face_landmarks.landmark:
                    embedding_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Преобразуем в numpy array
                landmarks_array = np.array(embedding_landmarks, dtype=np.float32)
                
                # Извлекаем landmarks для отрисовки (только 2D координаты)
                draw_landmarks = []
                h, w = image.shape[:2]
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    draw_landmarks.append((x, y))
                
                # Нормализуем эмбеддинг
                if len(landmarks_array) == self.embedding_size:
                    embedding = self._normalize_embedding(landmarks_array)
                    logger.debug("Face embedding and landmarks extracted successfully")
                    return embedding, draw_landmarks
            
            logger.warning("No face landmarks detected in image")
            return None, None
            
        except Exception as e:
            logger.error(f"Error extracting embedding with landmarks: {e}")
            return None, None

    def extract_embedding_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга из байтов изображения
        """
        try:
            # Конвертируем bytes в numpy array
            image = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image from bytes")
                return None
                
            return self.extract_embedding(image)
        except Exception as e:
            logger.error(f"Error extracting embedding from bytes: {e}")
            return None

    def extract_embedding_from_pil(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга из PIL Image
        """
        try:
            # Конвертируем PIL Image в numpy array
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return self.extract_embedding(image)
        except Exception as e:
            logger.error(f"Error extracting embedding from PIL image: {e}")
            return None

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Обнаружение всех лиц в изображении
        """
        try:
            if len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    if not detection.location_data:
                        continue
                        
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    # Конвертируем относительные координаты в абсолютные
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Добавляем отступы
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    width = min(w - x, width + 2 * padding)
                    height = min(h - y, height + 2 * padding)
                    
                    # Получаем уверенность
                    confidence = self.safe_float(detection.score[0] if detection.score else 0.0)
                    
                    face_info = {
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'keypoints': self._extract_keypoints(detection, w, h)
                    }
                    faces.append(face_info)
            
            logger.debug(f"Detected {len(faces)} faces in image")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[int], float, Optional[int]]:
        """
        Распознавание лица по эмбеддингу
        
        Returns:
            Tuple[person_id, confidence, photo_id]
        """
        if embedding is None or len(self.known_embeddings) == 0:
            return None, 0.0, None
        
        best_match_id = None
        best_photo_id = None
        best_similarity = 0.0
        
        for i, known_embedding in enumerate(self.known_embeddings):
            similarity = self._calculate_similarity(embedding, known_embedding)
            
            if similarity > best_similarity and similarity > self.recognition_threshold:
                best_similarity = similarity
                best_match_id = self.known_ids[i]
                best_photo_id = self.known_photo_ids[i]
        
        logger.debug(f"Recognition result: ID={best_match_id}, confidence={best_similarity:.3f}")
        return best_match_id, best_similarity, best_photo_id

    def recognize_face_in_image(self, image: np.ndarray, extract_landmarks: bool = False) -> List[Dict[str, Any]]:
        """
        Распознавание всех лиц в изображении
        
        Args:
            image: Входное изображение
            extract_landmarks: Извлекать landmarks для отрисовки
            
        Returns:
            Список результатов распознавания
        """
        results = []
        faces = self.detect_faces(image)
        
        for face in faces:
             x, y, w, h = face['bbox']
        if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
            face_roi = image[y:y+h, x:x+w]
            
            if extract_landmarks:
                # ИСПРАВЛЕНИЕ: extract_embedding_with_landmarks возвращает 2 значения, а не 3
                embedding, landmarks = self.extract_embedding_with_landmarks(face_roi)
                face['landmarks'] = landmarks
            else:
                embedding = self.extract_embedding(face_roi)
            
            # ИСПРАВЛЕНИЕ: recognize_face возвращает 3 значения
            person_id, confidence, photo_id = self.recognize_face(embedding)
            
            result = {
                'bbox': face['bbox'],
                'detection_confidence': face['confidence'],
                'person_id': person_id,
                'recognition_confidence': confidence,
                'photo_id': photo_id,
                'landmarks': face.get('landmarks')
            }
            results.append(result)
        
        return results

    def draw_landmarks(self, image: np.ndarray, landmarks: List, color: tuple = None, 
                      thickness: int = None, radius: int = None) -> np.ndarray:
        """
        Отрисовка landmarks на изображении
        """
        if not landmarks:
            return image
            
        try:
            color = color or self.landmark_style['color']
            thickness = thickness or self.landmark_style['thickness']
            radius = radius or self.landmark_style['radius']
            
            for (x, y) in landmarks:
                cv2.circle(image, (x, y), radius, color, thickness)
                
            return image
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return image

    def draw_face_connections(self, image: np.ndarray, landmarks: List, color: tuple = None, 
                            thickness: int = None) -> np.ndarray:
        """
        Отрисовка соединений между landmarks (контуры лица)
        """
        if not landmarks or len(landmarks) < 468:
            return image
            
        try:
            color = color or self.landmark_style['color']
            thickness = thickness or self.landmark_style['thickness']
            
            # Определяем важные контуры лица
            contours = [
                # Контур губ
                list(range(61, 68)) + [61],  # Верхняя губа
                list(range(67, 76)) + [67],  # Нижняя губа
                # Брови
                list(range(276, 283)),  # Левая бровь
                list(range(283, 290)),  # Правая бровь
                # Глаза
                list(range(469, 476)) + [469],  # Левый глаз
                list(range(476, 483)) + [476],  # Правый глаз
                # Нос
                list(range(1, 6)),  # Переносица
                list(range(6, 12))   # Кончик носа
            ]
            
            for contour in contours:
                for i in range(len(contour) - 1):
                    start_idx = contour[i]
                    end_idx = contour[i + 1]
                    
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = landmarks[start_idx]
                        end_point = landmarks[end_idx]
                        cv2.line(image, start_point, end_point, color, thickness)
            
            return image
        except Exception as e:
            logger.error(f"Error drawing face connections: {e}")
            return image

    def set_landmarks_style(self, color: tuple = None, thickness: int = None, radius: int = None):
        """Установка стиля отрисовки landmarks"""
        if color:
            self.landmark_style['color'] = color
        if thickness:
            self.landmark_style['thickness'] = thickness
        if radius:
            self.landmark_style['radius'] = radius

    def toggle_landmarks(self, show: bool = None):
        """Включение/отключение отображения landmarks"""
        if show is not None:
            self.show_landmarks = show
        else:
            self.show_landmarks = not self.show_landmarks
        logger.info(f"Landmarks display: {'enabled' if self.show_landmarks else 'disabled'}")

    # Остальные методы остаются без изменений...
    # (train_from_image, train_from_bytes, train_from_pil, add_existing_embedding, 
    # batch_train_person, load_embeddings_from_database, clear_embeddings, 
    # remove_embedding_by_photo_id, set_recognition_threshold, get_model_info, 
    # _add_embedding_to_memory, _normalize_embedding, _calculate_similarity, 
    # _extract_keypoints, draw_detection, cleanup)

    def train_from_image(self, image: np.ndarray, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Обучение модели на основе изображения
        """
        try:
            embedding = self.extract_embedding(image)
            if embedding is not None:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning("No face detected for training")
                return False
                
        except Exception as e:
            logger.error(f"Error training from image: {e}")
            return False

    def train_from_bytes(self, image_bytes: bytes, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Обучение модели на основе байтов изображения
        """
        try:
            embedding = self.extract_embedding_from_bytes(image_bytes)
            if embedding is not None:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning("No face detected for training from bytes")
                return False
        except Exception as e:
            logger.error(f"Error training from bytes: {e}")
            return False

    def train_from_pil(self, pil_image: Image.Image, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Обучение модели на основе PIL Image
        """
        try:
            embedding = self.extract_embedding_from_pil(pil_image)
            if embedding is not None:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning("No face detected for training from PIL image")
                return False
        except Exception as e:
            logger.error(f"Error training from PIL image: {e}")
            return False

    def add_existing_embedding(self, embedding: np.ndarray, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """
        Добавление существующего эмбеддинга в модель
        """
        try:
            if embedding is not None and len(embedding) == self.embedding_size:
                return self._add_embedding_to_memory(embedding, person_data, photo_id)
            else:
                logger.warning("Invalid embedding provided")
                return False
        except Exception as e:
            logger.error(f"Error adding existing embedding: {e}")
            return False

    def batch_train_person(self, person_id: int, person_name: str, database) -> int:
        """
        Пакетное обучение для одного человека по всем его фотографиям
        """
        try:
            trained_count = 0
            photos = database.get_person_photos(person_id)
            
            for photo in photos:
                if photo['id'] in self._embedding_cache:
                    # Используем кэшированный эмбеддинг
                    embedding = self._embedding_cache[photo['id']]
                    if self.add_existing_embedding(embedding, {'id': person_id, 'last_name': person_name}, photo['id']):
                        trained_count += 1
                elif photo.get('face_embedding'):
                    # Используем существующий эмбеддинг из базы
                    embedding = database.get_photo_embedding(photo['id'])
                    if embedding is not None:
                        if self.add_existing_embedding(embedding, {'id': person_id, 'last_name': person_name}, photo['id']):
                            # Кэшируем эмбеддинг
                            self._embedding_cache[photo['id']] = embedding
                            trained_count += 1
                else:
                    # Извлекаем эмбеддинг из изображения
                    pil_image = database.get_photo_as_image(photo['id'])
                    if pil_image:
                        if self.train_from_pil(pil_image, {'id': person_id, 'last_name': person_name}, photo['id']):
                            # Сохраняем эмбеддинг в базу
                            embedding = self.extract_embedding_from_pil(pil_image)
                            if embedding is not None:
                                database.update_photo_embedding(photo['id'], embedding)
                                # Кэшируем эмбеддинг
                                self._embedding_cache[photo['id']] = embedding
                            trained_count += 1
            
            logger.info(f"Batch trained {trained_count} photos for person {person_name}")
            return trained_count
            
        except Exception as e:
            logger.error(f"Error in batch training for person {person_id}: {e}")
            return 0

    def load_embeddings_from_database(self, database) -> int:
        """
        Загрузка эмбеддингов из базы данных
        """
        try:
            self.clear_embeddings()
            self._embedding_cache.clear()
            
            people = database.get_all_people()
            loaded_count = 0
            
            for person in people:
                photos = database.get_person_photos(person['id'])
                for photo in photos:
                    embedding = database.get_photo_embedding(photo['id'])
                    if embedding is not None:
                        if len(embedding) == self.embedding_size:
                            self.known_embeddings.append(embedding)
                            
                            # Создаем имя для отображения
                            last_name = person.get('last_name', '')
                            first_name = person.get('first_name', '')
                            display_name = f"{last_name} {first_name}".strip()
                            self.known_names.append(display_name)
                            
                            self.known_ids.append(person['id'])
                            self.known_photo_ids.append(photo['id'])
                            
                            # Кэшируем эмбеддинг
                            self._embedding_cache[photo['id']] = embedding
                            loaded_count += 1
                            
            logger.info(f"Loaded {loaded_count} embeddings from database")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading embeddings from database: {e}")
            return 0

    def clear_embeddings(self):
        """Очистка всех эмбеддингов из памяти"""
        self.known_embeddings.clear()
        self.known_names.clear()
        self.known_ids.clear()
        self.known_photo_ids.clear()
        logger.info("Cleared all embeddings from memory")

    def remove_embedding_by_photo_id(self, photo_id: int) -> bool:
        """
        Удаление эмбеддинга по ID фотографии
        """
        try:
            if photo_id in self.known_photo_ids:
                index = self.known_photo_ids.index(photo_id)
                self.known_embeddings.pop(index)
                self.known_names.pop(index)
                self.known_ids.pop(index)
                self.known_photo_ids.pop(index)
                
                # Удаляем из кэша
                if photo_id in self._embedding_cache:
                    del self._embedding_cache[photo_id]
                    
                logger.info(f"Removed embedding for photo ID: {photo_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing embedding for photo {photo_id}: {e}")
            return False

    def set_recognition_threshold(self, threshold: float):
        """Установка порога распознавания"""
        self.recognition_threshold = max(0.1, min(1.0, self.safe_float(threshold, 0.6)))
        logger.info(f"Recognition threshold set to {self.recognition_threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        unique_people = len(set(self.known_ids)) if self.known_ids else 0
        
        return {
            'loaded_embeddings': len(self.known_embeddings),
            'unique_people': unique_people,
            'recognition_threshold': self.recognition_threshold,
            'min_detection_confidence': self.min_detection_confidence,
            'embedding_size': self.embedding_size,
            'cache_size': len(self._embedding_cache),
            'show_landmarks': self.show_landmarks,
            'status': 'ready' if len(self.known_embeddings) > 0 else 'needs_training'
        }

    def _add_embedding_to_memory(self, embedding: np.ndarray, person_data: Dict[str, Any], photo_id: int = None) -> bool:
        """Добавление эмбеддинга в память модели"""
        self.known_embeddings.append(embedding)
        
        # Создаем имя для отображения
        last_name = person_data.get('last_name', '')
        first_name = person_data.get('first_name', '')
        display_name = f"{last_name} {first_name}".strip()
        self.known_names.append(display_name)
        
        # Используем существующий ID или создаем новый
        person_id = person_data.get('id')
        if person_id is None:
            person_id = len(self.known_ids) + 1
        self.known_ids.append(person_id)
        
        # Сохраняем ID фотографии
        self.known_photo_ids.append(photo_id)
        
        logger.info(f"Added embedding for {display_name}, photo ID: {photo_id}")
        return True

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Нормализация эмбеддинга"""
        mean = np.mean(embedding)
        std = np.std(embedding)
        
        if std > 0:
            normalized = (embedding - mean) / std
        else:
            normalized = embedding - mean
            
        return normalized

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисление схожести между эмбеддингами"""
        try:
            # Косинусное сходство
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Приводим к диапазону [0, 1]
            return float((similarity + 1) / 2)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _extract_keypoints(self, detection, image_width: int, image_height: int) -> List[Tuple[int, int]]:
        """Извлечение ключевых точек лица"""
        keypoints = []
        try:
            if hasattr(detection.location_data, 'relative_keypoints'):
                for keypoint in detection.location_data.relative_keypoints:
                    x = int(keypoint.x * image_width)
                    y = int(keypoint.y * image_height)
                    keypoints.append((x, y))
        except Exception as e:
            logger.warning(f"Error extracting keypoints: {e}")
            
        return keypoints

    def draw_detection(self, image: np.ndarray, face_info: Dict[str, Any], 
                      person_name: str = None, confidence: float = None) -> np.ndarray:
        """
        Отрисовка обнаруженного лица на изображении
        """
        try:
            x, y, w, h = face_info['bbox']
            
            # Выбираем цвет в зависимости от результата распознавания
            if person_name and confidence is not None:
                color = (0, 255, 0)  # Зеленый для распознанных
                conf_str = f"{self.safe_float(confidence):.2f}"
                label = f"{person_name} ({conf_str})"
            else:
                color = (0, 0, 255)  # Красный для неизвестных
                face_confidence = self.safe_float(face_info.get('detection_confidence', 0.0))
                conf_str = f"{face_confidence:.2f}"
                label = f"Unknown ({conf_str})"
            
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Рисуем подпись
            font_scale = 0.6
            thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            label_y = max(y - 10, label_size[1] + 10)
            
            # Фон для текста
            cv2.rectangle(image, (x, label_y - label_size[1] - 10), 
                         (x + label_size[0], label_y), color, -1)
            cv2.putText(image, label, (x, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Рисуем ключевые точки если есть
            for kp_x, kp_y in face_info.get('keypoints', []):
                cv2.circle(image, (kp_x, kp_y), 3, color, -1)
                
            return image
            
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
            return image

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            self._embedding_cache.clear()
            logger.info("KaleidoID resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")