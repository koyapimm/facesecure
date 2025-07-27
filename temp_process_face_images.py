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
