import tabula
import pandas as pd
import os
import requests
url="https://archive.ics.uci.edu/static/public/222/data.csv" 

try:
    # CSV'yi oku
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(url)
    
    # NaN değerlerini "NaN" olarak göster
    print("Tablo Örneği (ilk 5 satır):")
    print(df.head().to_string(na_rep="NaN"))
    
    # İstatistikler (NaN'lar "NaN" olarak)
    print("\nSayısal Verilerin İstatistikleri:")
    print(df.describe().to_string(na_rep="NaN"))
    
    # CSV'yi kaydet (NaN'lar "NaN" olarak)
    df.to_csv("verilerim.csv", index=False, na_rep="NaN")
    print("\nTablo 'verilerim.csv' olarak kaydedildi")

except requests.exceptions.RequestException as e:
    print(f"Bağlantı hatası: {e}")
except pd.errors.ParserError as e:
    print(f"CSV ayrıştırma hatası: {e}")
except Exception as e:
    print(f"Beklenmeyen hata: {e}")