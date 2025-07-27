import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
from dotenv import load_dotenv
import time

# Requests için yeniden deneme stratejisi
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

# API URL'i
API_URL = "http://127.0.0.1:5000"

def check_api_health():
    """API'nin çalışıp çalışmadığını kontrol et"""
    try:
        response = http.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# .env dosyasını yükle
load_dotenv()

def img_to_bytes(img):
    """Resmi byte dizisine çevirir"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img.save(tmp_file.name, format="JPEG")
        with open(tmp_file.name, "rb") as f:
            return f.read()

def login():
    """Admin girişi için login fonksiyonu"""
    if 'admin_token' not in st.session_state:
        st.session_state.admin_token = None
        
    if st.session_state.admin_token is None:
        # API sağlık kontrolü
        if not check_api_health():
            st.error("⚠️ API sunucusuna bağlanılamıyor! Lütfen sunucunun çalıştığından emin olun.")
            return False
            
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı:")
            password = st.text_input("Şifre:", type="password")
            submit = st.form_submit_button("Giriş Yap")
            
            if submit:
                try:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = http.post(
                                f"{API_URL}/login",
                                json={"username": username, "password": password},
                                timeout=10
                            )
                            if response.status_code == 200:
                                token = response.json()['token']
                                st.session_state.admin_token = token
                                st.success("Giriş başarılı!")
                                st.experimental_rerun()
                                break
                            else:
                                st.error("Giriş başarısız! Kullanıcı adı veya şifre hatalı.")
                                break
                        except requests.exceptions.RequestException as e:
                            if attempt == max_retries - 1:  # Son deneme
                                st.error(f"Bağlantı hatası: API sunucusuna ulaşılamıyor. Lütfen sunucunun çalıştığından emin olun.")
                            else:
                                time.sleep(1)  # 1 saniye bekle ve tekrar dene
                except Exception as e:
                    st.error(f"Beklenmeyen hata: {str(e)}")
        return False
    return True

st.set_page_config(page_title="FaceSecure", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠 FaceSecure - Yüz Tanıma Sistemi</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio("\n**İşlem Seç:**", ("Kayıt", "Tanıma"))

if menu == "Kayıt":
    st.subheader("📑 Yeni Kişi Kaydı")
    
    # Admin girişi kontrolü
    if not login():
        st.warning("Kayıt işlemi için admin girişi gerekli!")
        st.stop()

    camera = st.camera_input("Kamera Görüntüsü")
    if "images" not in st.session_state:
        st.session_state.images = []

    if st.button("📸 Fotoğrafı Kaydet"):
        if camera:
            img = Image.open(camera)
            st.session_state.images.append(img)

    if st.session_state.images:
        st.markdown("### 🖼️ Çekilen Fotoğraflar")
        new_images = []
        for idx, img in enumerate(st.session_state.images):
            cols = st.columns([1, 0.1])
            with cols[0]:
                st.image(img, width=150)
            with cols[1]:
                delete = st.button("❌ Sil", key=f"delete_{idx}")
            if not delete:
                new_images.append(img)
        st.session_state.images = new_images

    st.markdown("### 🧾 Kişi Bilgisi")
    name = st.text_input("👤 İsim girin:")
    
    # Fotoğraf sayısı kontrolü ve uyarı
    current_photos = len(st.session_state.images)
    if current_photos < 10:
        st.warning(f"⚠️ Şu an {current_photos}/10 fotoğraf çekildi. Kayıt için en az 10 fotoğraf gerekli!")
        st.progress(current_photos/10)
        
    if st.button("✅ Kaydı Gönder", disabled=current_photos < 10):
        if name and st.session_state.images:
            # En az 10 fotoğraf kontrolü
            if len(st.session_state.images) < 10:
                st.error("En az 10 fotoğraf gerekli!")
                st.stop()
                
            if not check_api_health():
                st.error("⚠️ API sunucusuna bağlanılamıyor! Lütfen sunucunun çalıştığından emin olun.")
                st.stop()

            try:
                files = [("images", (f"image_{i}.jpg", img_to_bytes(img), "image/jpeg")) for i, img in enumerate(st.session_state.images)]
                
                # JWT token'ı header'a ekle
                headers = {
                    'Authorization': f'Bearer {st.session_state.admin_token}'
                }
                
                with st.spinner('Kayıt işlemi yapılıyor... İlk kayıt uzun sürebilir. Lütfen bekleyin...'):
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = http.post(
                                f"{API_URL}/register", 
                                files=files, 
                                data={"name": name},
                                headers=headers,
                                timeout=200  # Kayıt işlemi uzun sürebilir
                            )
                
                            if response.status_code == 200:
                                st.success(f"✅ {name} başarıyla kaydedildi!")
                                # Kayıt başarılı olunca fotoğrafları temizle
                                st.session_state.images = []
                                break
                            elif response.status_code == 401:
                                st.error("⚠️ Oturum süresi dolmuş! Lütfen tekrar giriş yapın.")
                                st.session_state.admin_token = None
                                st.stop()
                            else:
                                error_msg = response.json().get('error', 'Bilinmeyen hata')
                                st.error(f"❌ Kayıt başarısız: {error_msg}")
                                break
                                
                        except requests.exceptions.RequestException as e:
                            if attempt == max_retries - 1:  # Son deneme
                                st.error("⚠️ API sunucusuna bağlanılamıyor! Lütfen internet bağlantınızı ve sunucunun çalıştığından emin olun.")
                            else:
                                time.sleep(1)  # 1 saniye bekle ve tekrar dene
                    
            except Exception as e:
                st.error(f"⚠️ Beklenmeyen hata: {str(e)}")
        else:
            if not name:
                st.warning("Lütfen isim girin.")
            if not st.session_state.images:
                st.warning("Lütfen en az 10 fotoğraf çekin.")


elif menu == "Tanıma":
    st.subheader("🔍 Kişi Tanıma")
    camera = st.camera_input("Tanıma için fotoğraf çekin")

    if st.button("🧠 Tanımayı Başlat"):
        if not camera:
            st.warning("Lütfen önce fotoğraf çekin.")
            st.stop()
            
        try:
            image = Image.open(camera)
            files = {"image": img_to_bytes(image)}
            response = requests.post(
                "http://127.0.0.1:5000/recognize", 
                files={"image": ("image.jpg", img_to_bytes(image), "image/jpeg")},
                headers={'Authorization': f'Bearer {st.session_state.admin_token or ""}'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    st.success(f"👤 Tanınan kişi: {result['name']} (Benzerlik: {(1-float(result['distance'])):.2%})")
                else:
                    error_msg = result.get('error', 'Kişi tanınamadı.')
                    if "multiple faces" in error_msg.lower():
                        st.error("⚠️ Birden fazla yüz tespit edildi! Lütfen sadece tek kişilik fotoğraf çekin.")
                    else:
                        st.warning(error_msg)
            elif response.status_code == 401:
                st.error("⚠️ Yetkilendirme hatası! Lütfen tekrar giriş yapın.")
                st.session_state.admin_token = None
            else:
                try:
                    error_msg = response.json().get('error', 'Sunucu hatası oluştu.')
                except:
                    error_msg = 'Sunucu hatası oluştu.'
                st.error(f"⚠️ Hata: {error_msg}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ API bağlantı hatası! Lütfen sunucunun çalıştığından emin olun.")
        except Exception as e:
            st.error(f"⚠️ Beklenmeyen hata: {str(e)}")



