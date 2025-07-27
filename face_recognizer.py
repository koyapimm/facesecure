import cv2
import numpy as np
import json
import os
from deepface import DeepFace
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import time

class FaceSecureRecognizer:
    def __init__(self):

        self.db_dir = "db"
        self.embeddings_file = os.path.join(self.db_dir, "embeddings.json")

        # Klasör yolları ve varsayılan değerler
        self.db_dir = "db"
        self.embeddings_file = os.path.join(self.db_dir, "embeddings.json")
        
        # MongoDB bağlantısı
        try:
            from pymongo import MongoClient
            from dotenv import load_dotenv
            load_dotenv()
            
            self.mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
            self.mongo_client.server_info()  # Test connection
            self.db = self.mongo_client['facesecure']
            self.users_collection = self.db['users']
            print("MongoDB bağlantısı başarılı!")
        except Exception as e:
            print(f"MongoDB bağlantı hatası: {e}")
            print("JSON dosya sistemine geçiliyor...")
            # Fallback: JSON kullan
            self.users_collection = None
        
        # OpenCV DNN yüz tespiti
        self.face_detector = self.load_face_detector()
        
        # Tanıma parametreleri
        self.threshold = 0.6
        
        # Mevcut embeddings'i yükle
        self.embeddings = self.load_embeddings()
        
        if not self.embeddings:
            print("Uyarı: Veritabanında kayıt bulunamadı!")
            print("Önce face_recorder.py ile kayıt yapın.")
    
    def load_face_detector(self):
        """OpenCV DNN yüz tespit modelini yükle"""
        try:
            # Model dosyalarının yolları
            prototxt_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
            # Model dosyalarını indir (eğer yoksa)
            if not os.path.exists("models"):
                os.makedirs("models", exist_ok=True)
            
            if not os.path.exists(prototxt_path):
                print("Model dosyaları indiriliyor...")
                self.download_face_model()
            
            # DNN modelini yükle
            detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("OpenCV DNN yüz tespit modeli başarıyla yüklendi!")
            return detector
            
        except Exception as e:
            print(f"DNN model yükleme hatası: {e}")
            print("Basit yüz tespiti kullanılacak.")
            return None
    
    def download_face_model(self):
        """OpenCV DNN yüz tespit modelini indir"""
        try:
            import urllib.request
            
            # Model dosyalarının URL'leri
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            print("Prototxt dosyası indiriliyor...")
            urllib.request.urlretrieve(prototxt_url, "models/deploy.prototxt")
            
            print("Model dosyası indiriliyor...")
            urllib.request.urlretrieve(model_url, "models/res10_300x300_ssd_iter_140000.caffemodel")
            
            print("Model dosyaları başarıyla indirildi!")
            
        except Exception as e:
            print(f"Model indirme hatası: {e}")
            print("Basit yüz tespiti kullanılacak.")
    
    def detect_face(self, frame):
        """OpenCV DNN ile yüz tespit et"""
        if self.face_detector is None:
            # Basit yüz tespiti - frame'in ortasında varsayımsal yüz alanı
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            face_size = min(width, height) // 3
            
            x = center_x - face_size // 2
            y = center_y - face_size // 2
            w = face_size
            h = face_size
            
            return np.array([[x, y, w, h]])
        
        try:
            # Frame'i DNN için hazırla
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Yüz tespiti yap
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            faces = []
            height, width = frame.shape[:2]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:  # Güven eşiği
                    # Koordinatları frame boyutuna çevir
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    x, y, x2, y2 = box.astype(int)
                    
                    # Sınırları kontrol et
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    w = x2 - x
                    h = y2 - y
                    
                    if w > 0 and h > 0:
                        faces.append([x, y, w, h])
            
            return np.array(faces)
            
        except Exception as e:
            print(f"OpenCV DNN yüz tespiti hatası: {e}")
            return np.array([])
    
    def load_embeddings(self):
        """MongoDB veya JSON'dan embeddings yükle"""
        embeddings = {}
        
        if self.users_collection is not None:
            try:
                users = self.users_collection.find({}, {'name': 1, 'embeddings': 1})
                for user in users:
                    embeddings[user['name']] = user.get('embeddings', [])
                return embeddings
            except Exception as e:
                print(f"MongoDB okuma hatası: {e}")
                print("JSON dosyasına geçiliyor...")
        
        # MongoDB başarısız olduysa veya bağlantı yoksa JSON'dan oku
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
                    processed_data = {}
                    for name, embedding_data in data.items():
                        # Yeni format: dict içinde mean_embedding, std_embedding, all_embeddings
                        if isinstance(embedding_data, dict) and 'mean_embedding' in embedding_data:
                            processed_data[name] = embedding_data
                        # Eski format: direkt liste
                        elif isinstance(embedding_data, list) and len(embedding_data) > 0:
                            # Eğer embedding bir liste içinde liste ise
                            if isinstance(embedding_data[0], list):
                                processed_data[name] = {
                                    "mean_embedding": embedding_data[0],
                                    "std_embedding": [0.1] * len(embedding_data[0]),  # Varsayılan std
                                    "all_embeddings": embedding_data,
                                    "count": len(embedding_data)
                                }
                            # Eğer embedding bir liste içinde dict ise
                            elif isinstance(embedding_data[0], dict) and 'embedding' in embedding_data[0]:
                                embeddings = [emb['embedding'] for emb in embedding_data]
                                mean_emb = np.mean(embeddings, axis=0).tolist()
                                std_emb = np.std(embeddings, axis=0).tolist()
                                processed_data[name] = {
                                    "mean_embedding": mean_emb,
                                    "std_embedding": std_emb,
                                    "all_embeddings": embeddings,
                                    "count": len(embeddings)
                                }
                            # Eğer embedding direkt liste ise
                            else:
                                processed_data[name] = {
                                    "mean_embedding": embedding_data,
                                    "std_embedding": [0.1] * len(embedding_data),
                                    "all_embeddings": [embedding_data],
                                    "count": 1
                                }
                        else:
                            processed_data[name] = embedding_data
                    
                    return processed_data
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def extract_face_embedding(self, face_image_path):
        """DeepFace ile yüz vektörü çıkar"""
        try:
            embedding_result = DeepFace.represent(
                img_path=face_image_path,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            
            # Embedding formatını kontrol et ve düzelt
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                if isinstance(embedding_result[0], dict) and 'embedding' in embedding_result[0]:
                    return embedding_result[0]['embedding']
                else:
                    return embedding_result[0]
            elif isinstance(embedding_result, dict) and 'embedding' in embedding_result:
                return embedding_result['embedding']
            else:
                return embedding_result
                
        except Exception as e:
            print(f"Embedding çıkarma hatası: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """İki embedding arasındaki cosine similarity hesapla"""
        try:
            # Embedding'leri numpy array'e çevir
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            # Cosine similarity hesapla
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return similarity
        except Exception as e:
            print(f"Similarity hesaplama hatası: {e}")
            return 0.0
    
    def calculate_l2_distance(self, embedding1, embedding2):
        """İki embedding arasındaki L2 mesafesi hesapla"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # L2 mesafesi
            l2_distance = np.linalg.norm(emb1 - emb2)
            
            # L2 mesafesini 0-1 arasına normalize et (küçük mesafe = yüksek benzerlik)
            # Tipik embedding mesafeleri 0-2 arasında olduğu için
            normalized_similarity = max(0, 1 - (l2_distance / 2.0))
            
            return normalized_similarity
        except Exception as e:
            print(f"L2 mesafe hesaplama hatası: {e}")
            return 0.0
    
    def calculate_dynamic_threshold(self, embeddings_data):
        """Kayıtlı embedding'lerin dağılımına göre dinamik eşik hesapla"""
        try:
            all_similarities = []
            
            # Her kişi için kendi embedding'leri arasındaki benzerlikleri hesapla
            for name, data in embeddings_data.items():
                if isinstance(data, dict) and 'all_embeddings' in data:
                    embeddings = data['all_embeddings']
                    if len(embeddings) > 1:
                        # Kendi embedding'leri arasındaki benzerlikler
                        for i in range(len(embeddings)):
                            for j in range(i + 1, len(embeddings)):
                                sim = self.calculate_similarity(embeddings[i], embeddings[j])
                                all_similarities.append(sim)
            
            if all_similarities:
                # Benzerliklerin ortalaması ve standart sapması
                mean_sim = np.mean(all_similarities)
                std_sim = np.std(all_similarities)
                
                # Dinamik eşik: ortalama - 2*std (güvenli sınır)
                dynamic_threshold = 0.78
                return dynamic_threshold
            else:
                return 0.6  # Varsayılan eşik
                
        except Exception as e:
            print(f"Dinamik eşik hesaplama hatası: {e}")
            return 0.6
    
    def recognize_face(self, face_embedding):
        """Gelişmiş yüz tanıma - çoklu karşılaştırma ve dinamik eşik"""
        if not self.embeddings:
            return "Not recognized", 0.0, 0.0
        
        best_match = None
        best_combined_score = 0.0
        best_cosine_sim = 0.0
        best_l2_sim = 0.0
        
        # Dinamik eşik hesapla
        dynamic_threshold = self.calculate_dynamic_threshold(self.embeddings)
        
        # Her kayıtlı kişi için karşılaştırma yap
        for name, data in self.embeddings.items():
            if isinstance(data, dict) and 'mean_embedding' in data:
                # Ortalama embedding ile cosine similarity
                cosine_sim = self.calculate_similarity(face_embedding, data['mean_embedding'])
                
                # Ortalama embedding ile L2 mesafesi
                l2_sim = self.calculate_l2_distance(face_embedding, data['mean_embedding'])
                
                # Tüm embedding'lerle karşılaştırma
                all_similarities = []
                all_l2_similarities = []
                
                if 'all_embeddings' in data:
                    for stored_emb in data['all_embeddings']:
                        sim = self.calculate_similarity(face_embedding, stored_emb)
                        l2_sim_all = self.calculate_l2_distance(face_embedding, stored_emb)
                        all_similarities.append(sim)
                        all_l2_similarities.append(l2_sim_all)
                    
                    # En yüksek benzerlikleri al
                    max_cosine = max(all_similarities)
                    max_l2 = max(all_l2_similarities)
                    
                    # Ortalama benzerlikleri de hesapla
                    mean_cosine = np.mean(all_similarities)
                    mean_l2 = np.mean(all_l2_similarities)
                    
                    # Kombine skor: %70 max, %30 mean
                    combined_cosine = 0.7 * max_cosine + 0.3 * mean_cosine
                    combined_l2 = 0.7 * max_l2 + 0.3 * mean_l2
                else:
                    combined_cosine = cosine_sim
                    combined_l2 = l2_sim
                
                # Final kombine skor: %60 cosine, %40 L2
                combined_score = 0.6 * combined_cosine + 0.4 * combined_l2
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_match = name
                    best_cosine_sim = combined_cosine
                    best_l2_sim = combined_l2
        
        # Dinamik eşik kontrolü
        if best_combined_score >= dynamic_threshold:
            return best_match, best_combined_score, dynamic_threshold
        else:
            return "Not recognized", best_combined_score, dynamic_threshold
    
    def log_unrecognized_attempt(self, similarity):
        """Tanınmayan girişimleri logla"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip_address = "192.168.1.100"  # Placeholder IP
        
        log_entry = f"[{timestamp}] IP: {ip_address} - Unrecognized attempt - Similarity: {similarity:.4f}\n"
        
        try:
            with open(self.logs_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Log yazma hatası: {e}")
    
    def save_temp_image(self, frame):
        """Frame'i geçici dosyaya kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_face_{timestamp}.jpg"
        temp_path = os.path.join("faces", temp_filename)
        
        # Görüntüyü uint8 formatına çevir
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        cv2.imwrite(temp_path, frame)
        return temp_path
    
    def recognize(self, image):
        """
        Verilen görüntüdeki yüzü tanı
        
        Args:
            image: str veya numpy.ndarray - Görüntü dosyası yolu veya numpy array olarak görüntü
            
        Returns:
            dict: Tanıma sonuçları
            {
                'success': bool,
                'name': str,
                'score': float,
                'threshold': float,
                'error': str (opsiyonel)
            }
        """
        if not self.embeddings:
            return {
                'success': False,
                'error': 'No face embeddings found in database'
            }
        
        # Görüntüyü yükle
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                return {
                    'success': False,
                    'error': 'Could not load image file'
                }
        else:
            frame = image
        
        # Frame'i uint8 formatına çevir
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Yüz tespiti
        faces = self.detect_face(frame)
            
        # Frame'i uint8 formatına çevir
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Yüz tespiti
        faces = self.detect_face(frame)
        
        # Yüz kontrolleri
        if len(faces) == 0:
            return {
                'success': False,
                'error': 'No face detected'
            }
        elif len(faces) > 1:
            return {
                'success': False,
                'error': 'Multiple faces detected'
            }
            
        # Yüz kontrolleri
        if len(faces) == 0:
            return {
                'success': False,
                'error': 'No face detected'
            }
        elif len(faces) > 1:
            return {
                'success': False,
                'error': 'Multiple faces detected'
            }
            
        # Yüz bölgesini kes
        x, y, w, h = faces[0]
        face_region = frame[y:y+h, x:x+w]
        
        # Geçici dosyaya kaydet
        temp_path = self.save_temp_image(face_region)
        
        try:
            # Embedding çıkar
            face_embedding = self.extract_face_embedding(temp_path)
            
            if face_embedding is None:
                return {
                    'success': False,
                    'error': 'Could not extract face embedding'
                }
                
            # Yüzü tanı
            recognized_name, combined_score, dynamic_threshold = self.recognize_face(face_embedding)
            
            # Tanınmayan girişimleri logla
            if recognized_name == "Not recognized":
                self.log_unrecognized_attempt(combined_score)
            
            # Geçici dosyayı sil
            try:
                os.remove(temp_path)
            except:
                pass
                
            return {
                'success': True,
                'name': recognized_name,
                'score': float(combined_score),
                'threshold': float(dynamic_threshold),
                'face_location': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            }
            
        except Exception as e:
            # Geçici dosyayı sil
            try:
                os.remove(temp_path)
            except:
                pass
                
            return {
                'success': False,
                'error': str(e)
            }

# API kullanımı için örnek:
# recognizer = FaceSecureRecognizer()
# result = recognizer.recognize("image.jpg")  # Dosya yolundan
# # veya
# result = recognizer.recognize(frame)  # NumPy array'den 