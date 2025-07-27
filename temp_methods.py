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
