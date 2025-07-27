import os
import logging
import warnings
import hashlib
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np
import cv2
from face_recognizer import FaceSecureRecognizer
from face_recorder import FaceSecureRecorder

warnings.filterwarnings('ignore')  # TensorFlow ve DeepFace uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow loglarını bastır
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# .env dosyasını yükle
load_dotenv()

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log handler'ı konsola yönlendir
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
jwt = JWTManager(app)

# MongoDB bağlantısı
users_collection = None
try:
    mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'), serverSelectionTimeoutMS=2000)
    # Test connection
    mongo_client.server_info()
    db = mongo_client['facesecure']
    users_collection = db['users']
    print("MongoDB bağlantısı başarılı!")
except Exception as e:
    print(f"MongoDB bağlantı hatası: {e}")
    print("JSON dosya sistemine geçiliyor...")
    # Fallback: JSON dosyası kullan
    if not os.path.exists('db'):
        os.makedirs('db', exist_ok=True)
    if not os.path.exists('db/embeddings.json'):
        with open('db/embeddings.json', 'w') as f:
            json.dump({}, f)

# FaceSecure sınıflarının örneklerini oluştur
recognizer = FaceSecureRecognizer()
recorder = FaceSecureRecorder()

def hash_embedding(embedding):
    """Embedding'i SHA256 ile hashle"""
    return hashlib.sha256(str(embedding).encode()).hexdigest()

def log_request(endpoint, status, details):
    """İstek loglarını kaydet"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {endpoint} - Status: {status} - {details}\n"
    
    try:
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Log yazma hatası: {e}")

def convert_image_to_array(file):
    """Form'dan gelen dosyayı NumPy array'e çevir"""
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        return None

@app.route('/')
def index():
    """Ana sayfa"""
    return jsonify({
        "name": "FaceSecure API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "API health check",
            "POST /login": "Login to get JWT token",
            "POST /register": "Register a new face (JWT required)",
            "POST /recognize": "Recognize a face (JWT required)",
            "DELETE /delete_user": "Delete a user (JWT required)"
        }
    })

@app.route('/login', methods=['POST'])
def login():
    """Login endpoint'i"""
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    if username != os.getenv('ADMIN_USERNAME') or password != os.getenv('ADMIN_PASSWORD'):
        return jsonify({
            "success": False,
            "error": "Invalid credentials"
        }), 401
    
    access_token = create_access_token(identity=username)
    return jsonify({
        "success": True,
        "token": access_token
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü endpoint'i"""
    log_request("/health", "OK", "Health check request")
    return jsonify({
        "status": "FaceSecure API running"
    })

@app.route('/register', methods=['POST'])
@jwt_required()
def register():
    """Yüz kayıt endpoint'i"""
    try:
        # JWT kontrolü
        current_user = get_jwt_identity()
        if current_user != os.getenv('ADMIN_USERNAME'):
            return jsonify({
                "success": False,
                "error": "Unauthorized"
            }), 403
            
        # İsim kontrolü
        if 'name' not in request.form:
            log_request("/register", "ERROR", "Missing name field")
            return jsonify({
                "success": False,
                "error": "Name field is required"
            }), 400
            
        name = request.form['name'].strip()
        if not name:
            log_request("/register", "ERROR", "Empty name field")
            return jsonify({
                "success": False,
                "error": "Name cannot be empty"
            }), 400
            
        # Resim dosyaları kontrolü
        if 'images' not in request.files:
            log_request("/register", "ERROR", "No image files")
            return jsonify({
                "success": False,
                "error": "No image files uploaded"
            }), 400
            
        files = request.files.getlist('images')
        if len(files) < recorder.min_images:
            log_request("/register", "ERROR", f"Not enough images, minimum {recorder.min_images} required")
            return jsonify({
                "success": False,
                "error": f"At least {recorder.min_images} images required"
            }), 400
            
        images = []
        for file in files:
            image = convert_image_to_array(file)
            if image is None:
                log_request("/register", "ERROR", "Invalid image file")
                return jsonify({
                    "success": False,
                    "error": "Invalid image file"
                }), 400
            images.append(image)
            
        # Yüz kayıt işlemi
        embeddings = recorder.process_face_images(images, name)
        if not embeddings:
            log_request("/register", "ERROR", "Face detection failed")
            return jsonify({
                "success": False,
                "error": "Could not detect face in images"
            }), 400

        # Embedding'leri hashleme
        hashed_embeddings = [hash_embedding(emb) for emb in embeddings]
            
        # MongoDB'ye kaydet
        if users_collection is not None:
            user_data = {
                "name": name,
                "embeddings": hashed_embeddings,
                "mean_embedding": recorder.calculate_mean_embedding(embeddings).tolist(),
                "std_embedding": recorder.calculate_std_embedding(embeddings).tolist(),
                "count": len(embeddings),
                "created_at": datetime.now()
            }
            try:
                users_collection.insert_one(user_data)
            except Exception as e:
                print(f"MongoDB kayıt hatası: {e}")
                # Fallback: JSON'a kaydet
                recorder.save_embeddings()
        
        log_request("/register", "SUCCESS", f"User {name} registered with {len(embeddings)} images")
        return jsonify({
            "success": True,
            "message": f"User {name} registered successfully!"
        })
            
    except Exception as e:
        log_request("/register", "ERROR", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/recognize', methods=['POST'])
@jwt_required()
def recognize():
    """Yüz tanıma endpoint'i"""
    try:
        if 'image' not in request.files:
            log_request("/recognize", "ERROR", "No image file")
            return jsonify({
                "success": False,
                "error": "No image file uploaded"
            }), 400
            
        image = convert_image_to_array(request.files['image'])
        if image is None:
            log_request("/recognize", "ERROR", "Invalid image file")
            return jsonify({
                "success": False,
                "error": "Invalid image file"
            }), 400
            
        # Tanıma işlemi
        name, distance, error = recognizer.recognize_face(image)
        
        if error:
            log_request("/recognize", "ERROR", error)
            return jsonify({
                "success": False,
                "error": error
            }), 400
            
        if name:
            log_request("/recognize", "SUCCESS", f"Face recognized as {name}")
            return jsonify({
                "success": True,
                "name": name,
                "distance": float(distance)
            })
        else:
            log_request("/recognize", "WARNING", "Face not recognized")
            return jsonify({
                "success": False,
                "error": "Face not recognized"
            })
            
    except Exception as e:
        log_request("/recognize", "ERROR", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_user', methods=['DELETE'])
@jwt_required()
def delete_user():
    """Kullanıcı silme endpoint'i"""
    try:
        # JWT kontrolü
        current_user = get_jwt_identity()
        if current_user != os.getenv('ADMIN_USERNAME'):
            return jsonify({
                "success": False,
                "error": "Unauthorized"
            }), 403
            
        name = request.args.get('name')
        if not name:
            return jsonify({
                "success": False,
                "error": "Name parameter is required"
            }), 400
            
        # MongoDB'den sil
        deleted = False
        if users_collection is not None:
            try:
                result = users_collection.delete_one({"name": name})
                deleted = result.deleted_count > 0
            except Exception as e:
                print(f"MongoDB silme hatası: {e}")
                
        if not deleted:
            # JSON dosyasından silmeyi dene
            if name in recorder.embeddings:
                del recorder.embeddings[name]
                recorder.save_embeddings()
        
        # Yüz görsellerini sil
        user_face_dir = os.path.join(recorder.faces_dir, name)
        if os.path.exists(user_face_dir):
            import shutil
            shutil.rmtree(user_face_dir)
        
        log_request("/delete_user", "SUCCESS", f"User {name} deleted")
        return jsonify({
            "success": True,
            "message": f"User {name} deleted successfully"
        })
            
    except Exception as e:
        log_request("/delete_user", "ERROR", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

@app.route('/register', methods=['POST'])
def register():
    """Yüz kayıt endpoint'i"""
    try:
        # İsim kontrolü
        if 'name' not in request.form:
            log_request("/register", "ERROR", "Missing name field")
            return jsonify({
                "success": False,
                "error": "Name field is required"
            }), 400
            
        name = request.form['name'].strip()
        if not name:
            log_request("/register", "ERROR", "Empty name field")
            return jsonify({
                "success": False,
                "error": "Name cannot be empty"
            }), 400
            
        # Resim dosyaları kontrolü
        if 'images' not in request.files:
            log_request("/register", "ERROR", "No image files provided")
            return jsonify({
                "success": False,
                "error": "Image files are required"
            }), 400
            
        images = request.files.getlist('images')
        
        # Resim sayısı kontrolü
        if len(images) < 3:
            log_request("/register", "ERROR", "Too few images")
            return jsonify({
                "success": False,
                "error": "At least 3 images are required"
            }), 400
            
        if len(images) > 10:
            log_request("/register", "ERROR", "Too many images")
            return jsonify({
                "success": False,
                "error": "Maximum 10 images allowed"
            }), 400
            
        # Resimleri NumPy array'e çevir
        image_arrays = []
        for img in images:
            array = convert_image_to_array(img)
            if array is None:
                log_request("/register", "ERROR", f"Invalid image: {img.filename}")
                return jsonify({
                    "success": False,
                    "error": f"Could not process image: {img.filename}"
                }), 400
            image_arrays.append(array)
            
        # Kayıt işlemini yap
        result = recorder.register(name, image_arrays)
        
        if result['success']:
            log_request("/register", "SUCCESS", 
                       f"Name: {name}, Images: {result['images_processed']}, "
                       f"Embeddings: {result['embeddings_created']}")
            
            return jsonify({
                "success": True,
                "name": name,
                "images_saved": result['images_processed'],
                "embeddings_created": result['embeddings_created']
            })
        else:
            log_request("/register", "ERROR", result.get('error', 'Unknown error'))
            return jsonify({
                "success": False,
                "error": result.get('error', 'Registration failed')
            }), 400
            
    except Exception as e:
        log_request("/register", "ERROR", str(e))
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    """Yüz tanıma endpoint'i"""
    try:
        # Resim kontrolü
        if 'image' not in request.files:
            log_request("/recognize", "ERROR", "No image file provided")
            return jsonify({
                "success": False,
                "error": "Image file is required"
            }), 400
            
        image = request.files['image']
        
        # Resmi NumPy array'e çevir
        array = convert_image_to_array(image)
        if array is None:
            log_request("/recognize", "ERROR", f"Invalid image: {image.filename}")
            return jsonify({
                "success": False,
                "error": "Could not process image"
            }), 400
            
        # Tanıma işlemini yap
        result = recognizer.recognize(array)
        
        if result['success']:
            log_request("/recognize", "SUCCESS", 
                       f"Recognized as: {result['name']}, Score: {result['score']:.4f}")
            
            return jsonify({
                "success": True,
                "recognized_as": result['name'],
                "score": round(float(result['score']), 4),
                "threshold": round(float(result['threshold']), 4)
            })
        else:
            log_request("/recognize", "ERROR", result.get('error', 'Recognition failed'))
            return jsonify({
                "success": False,
                "error": result.get('error', 'Recognition failed')
            }), 400
            
    except Exception as e:
        log_request("/recognize", "ERROR", str(e))
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

if __name__ == "__main__":
    # Klasörlerin varlığını kontrol et
    os.makedirs("faces", exist_ok=True)
    os.makedirs("db", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # API'yi başlat
    app.run(host='0.0.0.0', port=5000, debug=True)
