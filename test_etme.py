import pandas as pd
import joblib

# Kullanılacak modeller ve etiketleri
modeller = {
    "Euclidean (Öklid)": 'knn_model_euclidean.pkl',
    "Manhattan (L1)": 'knn_model_manhattan.pkl',
    "Chebyshev (L∞)": 'knn_model_chebyshev.pkl'
}

# Müşteri örnekleri
potansiyel_musteriler = pd.DataFrame([
    {'age': 42, 'job': 'management', 'marital': 'married', 'education': 'tertiary', 
     'default': 'no', 'balance': 8500, 'housing': 'yes', 'loan': 'no', 
     'contact': 'cellular', 'day_of_week': 15, 'month': 'may', 'duration': 450, 
     'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'},

    {'age': 65, 'job': 'retired', 'marital': 'married', 'education': 'secondary', 
     'default': 'no', 'balance': 12000, 'housing': 'no', 'loan': 'no', 
     'contact': 'cellular', 'day_of_week': 10, 'month': 'jun', 'duration': 320, 
     'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'},

    {'age': 35, 'job': 'technician', 'marital': 'single', 'education': 'tertiary', 
     'default': 'no', 'balance': 5600, 'housing': 'yes', 'loan': 'no', 
     'contact': 'cellular', 'day_of_week': 3, 'month': 'apr', 'duration': 280, 
     'campaign': 2, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'},

    {'age': 52, 'job': 'self-employed', 'marital': 'married', 'education': 'secondary', 
     'default': 'no', 'balance': 7200, 'housing': 'yes', 'loan': 'no', 
     'contact': 'cellular', 'day_of_week': 20, 'month': 'jul', 'duration': 380, 
     'campaign': 1, 'pdays': 180, 'previous': 1, 'poutcome': 'success'},

    {'age': 29, 'job': 'management', 'marital': 'single', 'education': 'tertiary', 
     'default': 'no', 'balance': 6800, 'housing': 'no', 'loan': 'no', 
     'contact': 'cellular', 'day_of_week': 12, 'month': 'feb', 'duration': 310, 
     'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'}
])

# Tahminleri tüm modeller için yap
print("== Vadeli Mevduat Tahmin Karşılaştırması ==\n")

for i, (indeks, musteri) in enumerate(potansiyel_musteriler.iterrows()):
    print(f"\n--- Müşteri {i+1} ---")
    print(f"Profil ➤ Yaş: {musteri['age']}, Meslek: {musteri['job']}, Eğitim: {musteri['education']}, Bakiye: {musteri['balance']} €")

    for ad, dosya in modeller.items():
        try:
            model = joblib.load(dosya)
            tahmin = model.predict(musteri.to_frame().T)[0]
            olasiliklar = model.predict_proba(musteri.to_frame().T)[0]

            etik = "EVET - Açabilir" if tahmin == 1 else "HAYIR - Açmayabilir"
            guven = olasiliklar[1] * 100  # 'yes' sınıfı
            print(f"{ad} ➤ Tahmin: {etik} | Güven: %{guven:.2f}")
        except Exception as e:
            print(f"{ad} ➤ Tahmin yapılamadı: {e}")

# Genel istatistik vermek ister misin? Ekleyebilirim.
print("\n\n== GENEL İSTATİSTİKLER ==")
for ad, dosya in modeller.items():
    try:
        model = joblib.load(dosya)
        tahminler = model.predict(potansiyel_musteriler)
        toplam = len(tahminler)
        evet_sayisi = sum(1 for t in tahminler if t == 1)
        oran = (evet_sayisi / toplam) * 100
        print(f"{ad} ➤ {toplam} müşteriden {evet_sayisi} tanesi için 'Vadeli Mevduat Açabilir' dedi. (%{oran:.2f})")
    except Exception as e:
        print(f"{ad} ➤ İstatistik hesaplanamadı: {e}")