FROM python:3.11-slim

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.deepface/weights

# Çalışma dizinini oluştur
WORKDIR /app

# requirements.txt'yi kopyala ve bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Gerekli klasörleri oluştur
RUN mkdir -p faces db models

# Environment variables
ENV MONGO_URI=mongodb://localhost:27017/facesecure
ENV JWT_SECRET_KEY=your-super-secret-key
ENV ADMIN_USERNAME=admin
ENV ADMIN_PASSWORD=1234

# Portları aç
EXPOSE 5000 8501

# Flask ve Streamlit uygulamalarını başlat
CMD ["sh", "-c", "python app.py & streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0"]
