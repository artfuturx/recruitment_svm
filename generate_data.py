# --------------------------------------------

# generate_data.py

# Bu dosyada model eğitimi için basit ve eşit dağılmış sahte veriler üretilmektedir.
# 'years_experience' (0–10) ve 'technical_score' (0–100) değerleri random.uniform ile üretilir.
# Değerlendirme kuralı:
# - Eğer adayın deneyimi 2 yıldan az ve teknik puanı 60'tan düşükse --> label = 1 (Not Hired)
# - Diğer tüm durumlarda --> label = 0 (Hired)
# Ancak bu dağılım gerçek hayattaki gibi doğal değildir.

# Projede kullanılan veri seti, daha gerçekçi ve doğal dağılıma sahip olduğu için 
# generate_data_faker.py dosyasında üretilen veriler baz alınarak oluşturulmuştur.

# OUTPUT: candidates_data.csv

# --------------------------------------------


import pandas as pd
import numpy as np
import random

def generate_candidates(n=200):
    data = []
    
    for _ in range(n):
        years_experience = round(random.uniform(0, 10), 1)
        technical_score = round(random.uniform(0, 100), 1)
        
        if years_experience < 2 and technical_score < 60:
            label = 1  # Başvuru Reddedildi
        else:
            label = 0  # Başvuru Kabul Edildi
            
        data.append({
            'years_experience': years_experience,
            'technical_score': technical_score,
            'label': label
        })
    
    return pd.DataFrame(data)

candidates_df = generate_candidates(200)

candidates_df.to_csv('candidates_data.csv', index=False)

print(candidates_df.head())
print(candidates_df['label'].value_counts())