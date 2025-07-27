# Face Recognition System

Bu proje, yüz tanıma teknolojisini kullanarak güvenli kimlik doğrulama sistemi sunan bir uygulamadır. Flask API backend'i ve Streamlit frontend'i ile geliştirilmiştir.

## Özellikler

- Gerçek zamanlı yüz tespiti ve tanıma
- MongoDB veritabanı entegrasyonu (JSON yedekleme ile)
- JWT bazlı kimlik doğrulama
- Çoklu yüz kaydı desteği
- Ortalama ve standart sapma bazlı yüz doğrulama
- Kullanıcı dostu Streamlit arayüzü

## Gereksinimler

- Python 3.11+
- OpenCV
- DeepFace
- Flask
- Streamlit
- MongoDB (opsiyonel)
- NumPy
- dotenv

## Kurulum

1. Sanal ortam oluşturun ve aktif edin:
```bash
python -m venv faceenv
# Windows
faceenv\Scripts\activate
# Linux/Mac
source faceenv/bin/activate
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. MongoDB bağlantısı için .env dosyası oluşturun:
```
MONGODB_URI=your_mongodb_connection_string
```

## Kullanım

### API Sunucusunu Başlatma
```bash
python app.py
```

### Streamlit Arayüzünü Başlatma
```bash
streamlit run app_streamlit.py
```

## API Endpoints

- `POST /register`: Yeni yüz kaydı
- `POST /verify`: Yüz doğrulama
- `GET /users`: Kayıtlı kullanıcıları listeleme

## Güvenlik Özellikleri

- JWT bazlı kimlik doğrulama
- Yüz embeddingler için ortalama ve standart sapma hesaplaması
- MongoDB veya JSON dosya tabanlı güvenli veri depolama

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik branch'i oluşturun (`git checkout -b feature/amazing_feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik: Muhteşem özellik'`)
4. Branch'inizi push edin (`git push origin feature/amazing_feature`)
5. Bir Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.
