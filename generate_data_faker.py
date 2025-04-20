# --------------------------------------------------------

# generate_data_faker.py

# Bu dosyada daha gerçekçi bir veri seti oluşturmak amaçlanmıştır.
# 'Faker' kütüphanesi ile isimler, 'random.gauss' ile teknik skorlar üretilmektedir.
# Teknik skorlar 60 ortalama ve 20 standart sapma ile doğal bir dağılım gösterir.
# - Ancak gauss fonksiyonu nadiren 0'dan küçük veya 100'den büyük değer verebilir.
# - Bu yüzden skorlar 0–100 aralığında sınırlandırılmıştır.

# Değerlendirme kuralı:
# - Eğer adayın deneyimi 2 yıldan az ve teknik puanı 60'tan düşükse --> label = 1 (Not Hired)
# - Diğer tüm durumlarda --> label = 0 (Hired)

# Bu veri seti, projenin model eğitiminde kullanılmış ana veri kaynağıdır.
# Gerçekçiliği artırmak amacıyla generate_data.py yerine bu dosya tercih edilmiştir.

# OUTPUT: candidates_data_faker.csv

# --------------------------------------------------------


import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

def generate_candidates_faker(n=200):
    data = []

    for _ in range(n):
        name = fake.name()
        years_experience = round(random.uniform(0, 10), 1)
        technical_score = round(random.gauss(60, 20), 1) 
        technical_score = max(0, min(technical_score, 100))  

        if years_experience < 2 and technical_score < 60:
            label = 1
        else:
            label = 0

        data.append({
            'name': name,
            'years_experience': years_experience,
            'technical_score': technical_score,
            'label': label
        })

    return pd.DataFrame(data)

df = generate_candidates_faker(200)
df.to_csv('candidates_data_faker.csv', index=False)

print(df.head())
print(df['label'].value_counts())