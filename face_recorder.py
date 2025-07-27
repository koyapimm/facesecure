import cv2
import numpy as np
import json
import os
from deepface import DeepFace
from datetime import datetime
import time

class FaceSecureRecorder:
    def __init__(self):
        # Klasör yolları
        self.faces_dir = "faces"
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
            os.makedirs(self.db_dir, exist_ok=True)
        
        # OpenCV DNN yüz tespiti
        self.face_detector = self.load_face_detector()
        
        # Kayıt parametreleri
        self.min_images = 10
        self.face_detection_confidence = 0.3
        
        # Klasörleri oluştur
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Mevcut embeddings'i yükle
        self.embeddings = self.load_embeddings()
    
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
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"JSON okuma hatası: {e}")
                return {}
        return {}
    
    def save_embeddings(self):
        """Embeddings'i MongoDB veya JSON olarak kaydet"""
        try:
            if self.users_collection is not None:
                # MongoDB'ye kaydet
                for name, embeddings in self.embeddings.items():
                    self.users_collection.update_one(
                        {"name": name},
                        {
                            "$set": {
                                "embeddings": embeddings,
                                "updated_at": datetime.now()
                            }
                        },
                        upsert=True
                    )
            else:
                # JSON'a kaydet
                with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                    json.dump(self.embeddings, f, indent=4)
            return True
        except Exception as e:
            print(f"Embeddings kaydetme hatası: {e}")
            return False
    
    def process_face_images(self, images, name):
        """Gelen görüntülerden yüz tespiti yap ve embeddingler oluştur"""
        if len(images) < self.min_images:
            raise ValueError(f"En az {self.min_images} fotoğraf gerekli!")

        # Kullanıcı klasörünü oluştur
        user_dir = os.path.join(self.faces_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        embeddings = []
        saved_images = []

        for i, image in enumerate(images):
            # Yüz tespiti yap
            faces = self.detect_face(image)
            
            if len(faces) == 0:
                print(f"Uyarı: {i+1}. fotoğrafta yüz tespit edilemedi!")
                continue
                
            if len(faces) > 1:
                print(f"Uyarı: {i+1}. fotoğrafta birden fazla yüz tespit edildi!")
                continue
            
            # Yüz bölgesini kes
            x, y, w, h = faces[0]
            face_image = image[y:y+h, x:x+w]
            
            # Yüz görselini kaydet
            image_path = os.path.join(user_dir, f"{int(time.time())}_{i}.jpg")
            cv2.imwrite(image_path, face_image)
            saved_images.append(image_path)
            
            try:
                # DeepFace ile embedding çıkar
                embedding_obj = DeepFace.represent(
                    image_path,
                    model_name="VGG-Face",
                    enforce_detection=False
                )
                
                if embedding_obj:
                    embedding = embedding_obj[0]["embedding"]
                    embeddings.append(embedding)
                
            except Exception as e:
                print(f"Embedding çıkarma hatası ({i+1}. fotoğraf): {e}")
                continue

        if len(embeddings) < self.min_images:
            # Yetersiz embedding durumunda kaydedilen dosyaları temizle
            for img_path in saved_images:
                try:
                    os.remove(img_path)
                except:
                    pass
            os.rmdir(user_dir)
            raise ValueError(f"Yeterli kalitede yüz tespit edilemedi! En az {self.min_images} temiz fotoğraf gerekli.")

        # Embeddings'i kaydet
        self.embeddings[name] = embeddings
        self.save_embeddings()
        
        return embeddings

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
    
    def save_face_image(self, face_image, name, index):
        """Yüz görüntüsünü kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{index:02d}_{timestamp}.jpg"
        filepath = os.path.join(self.faces_dir, filename)
        
        # Görüntüyü uint8 formatına çevir
        if face_image.dtype != np.uint8:
            face_image = np.clip(face_image, 0, 255).astype(np.uint8)
        
        cv2.imwrite(filepath, face_image)
        return filepath
    
    def calculate_mean_embedding(self, embeddings):
        """Verilen embeddinglerde ortalama hesapla"""
        if not embeddings:
            return None
        # Embedding'leri numpy array'e çevir
        embeddings_array = np.array(embeddings)
        # Ortalama hesapla
        return np.mean(embeddings_array, axis=0)

    def calculate_std_embedding(self, embeddings):
        """Verilen embeddinglerde standart sapma hesapla"""
        if not embeddings:
            return None
        # Embedding'leri numpy array'e çevir
        embeddings_array = np.array(embeddings)
        # Standart sapma hesapla
        return np.std(embeddings_array, axis=0)
            
    def register(self, name, images):
        """
        Verilen görüntülerden yüz kaydı yap
        
        Args:
            name: str - Kaydedilecek kişinin adı
            images: list - Görüntü dosyası yolları veya numpy array olarak görüntüler listesi
            
        Returns:
            dict: Kayıt sonuçları
            {
                'success': bool,
                'message': str,
                'images_processed': int,
                'embeddings_created': int,
                'error': str (opsiyonel)
            }
        """
        if not name or not isinstance(name, str):
            return {
                'success': False,
                'message': 'Invalid name provided',
                'images_processed': 0,
                'embeddings_created': 0,
                'error': 'Name must be a non-empty string'
            }
            
        if not images or not isinstance(images, list):
            return {
                'success': False,
                'message': 'No images provided',
                'images_processed': 0,
                'embeddings_created': 0,
                'error': 'Images must be provided as a list'
            }
        
        embeddings = []
        processed_images = 0
        successful_embeddings = 0
        
        for img in images:
            processed_images += 1
            try:
                # Görüntüyü yükle
                if isinstance(img, str):
                    frame = cv2.imread(img)
                    if frame is None:
                        continue
                else:
                    frame = img
                    
                # Frame'i uint8 formatına çevir
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Yüz tespiti
                faces = self.detect_face(frame)
                
                if len(faces) != 1:
                    continue
                    
                # Yüz bölgesini kes
                x, y, w, h = faces[0]
                face_region = frame[y:y+h, x:x+w]
                
                # Geçici olarak kaydet
                face_path = self.save_face_image(face_region, name, successful_embeddings + 1)
                
                # Embedding çıkar
                embedding = self.extract_face_embedding(face_path)
                
                if embedding:
                    embeddings.append(embedding)
                    successful_embeddings += 1
                
                # Geçici dosyayı sil
                try:
                    os.remove(face_path)
                except:
                    pass
                    
            except Exception as e:
                continue
            
        # Yeterli embedding oluşturuldu mu kontrol et
        if successful_embeddings > 0:
            # Tüm embedding'leri kaydet
            self.embeddings[name] = embeddings
            self.save_embeddings()
            
            return {
                'success': True,
                'message': 'Registration successful',
                'images_processed': processed_images,
                'embeddings_created': successful_embeddings
            }
        else:
            return {
                'success': False,
                'message': 'No valid faces found in images',
                'images_processed': processed_images,
                'embeddings_created': 0,
                'error': 'Could not extract any valid face embeddings'
            }

# API kullanımı için örnek:
# recorder = FaceSecureRecorder()
# images = ["image1.jpg", "image2.jpg", ...]  # Dosya yolları listesi
# # veya
# images = [frame1, frame2, ...]  # NumPy array listesi
# result = recorder.register("John Doe", images) 