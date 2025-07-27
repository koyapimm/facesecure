#!/usr/bin/env python3
"""
Embeddings.json dosyasını yeni formata dönüştürme scripti
"""

import json
import numpy as np
import os

def convert_embeddings():
    """Mevcut embeddings.json dosyasını yeni formata dönüştür"""
    
    embeddings_file = "db/embeddings.json"
    
    if not os.path.exists(embeddings_file):
        print("embeddings.json dosyası bulunamadı!")
        return
    
    try:
        # Mevcut dosyayı oku
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Mevcut kayıtlar: {len(data)}")
        
        # Yeni format için dönüştürme
        converted_data = {}
        
        for name, embedding_data in data.items():
            print(f"İşleniyor: {name}")
            
            if isinstance(embedding_data, list) and len(embedding_data) > 0:
                # Embedding'leri numpy array'e çevir
                embeddings_array = np.array(embedding_data)
                
                # Ortalama embedding hesapla
                mean_embedding = np.mean(embeddings_array, axis=0).tolist()
                
                # Standart sapma hesapla
                std_embedding = np.std(embeddings_array, axis=0).tolist()
                
                # Yeni format
                converted_data[name] = {
                    "mean_embedding": mean_embedding,
                    "std_embedding": std_embedding,
                    "all_embeddings": embedding_data,
                    "count": len(embedding_data)
                }
                
                print(f"  - {len(embedding_data)} embedding dönüştürüldü")
            else:
                print(f"  - Geçersiz format, atlandı")
        
        # Yedek oluştur
        backup_file = embeddings_file + ".backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Yedek oluşturuldu: {backup_file}")
        
        # Yeni formatı kaydet
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"Dönüştürme tamamlandı! {len(converted_data)} kayıt güncellendi.")
        
    except Exception as e:
        print(f"Dönüştürme hatası: {e}")

if __name__ == "__main__":
    print("=== Embeddings Dönüştürme Scripti ===")
    convert_embeddings() 