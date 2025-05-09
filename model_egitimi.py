import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Veri setini yükle
df = pd.read_csv("verilerim.csv")  # kendi dosyanın adı

# Hedef değişkeni dönüştür (yes/no → 1/0)
le = LabelEncoder()
y = le.fit_transform(df["y"])

# Özellikleri ayır
X = df.drop("y", axis=1)
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Ön işleme pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Eğitim ve test seti ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mesafe metrikleri
metrics = {
    "euclidean": {"name": "Euclidean (p=2)", "params": {"metric": "minkowski", "p": 2}},
    "manhattan": {"name": "Manhattan (p=1)", "params": {"metric": "minkowski", "p": 1}},
    "chebyshev": {"name": "Chebyshev (L∞)", "params": {"metric": "chebyshev"}}
}

# K aralığı belirle
k_range = range(1, 21)

# Tüm metriklerin sonuçlarını saklayacak sözlük
metrics_results = {}

# Tüm K değerlerinin doğruluk skorlarını saklayacak sözlük
all_k_scores = {}

# Specificity dahil tüm metrikleri hesaplayan fonksiyon
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape[0] == 2:  # İkili sınıflandırma durumunda
        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:  # Çok sınıflı durumda 
        # Her sınıf için specificity hesapla ve ortalamasını al
        specificities = []
        n_classes = len(np.unique(y_true))
        
        for i in range(n_classes):
            # i sınıfı için true negative'leri hesapla
            # Karmaşıklık matrisinden i. satır ve i. sütunu çıkar
            mask = np.ones(cm.shape, bool)
            mask[i, :] = False
            mask[:, i] = False
            true_neg = np.sum(cm[mask])
            
            # i sınıfı için false positive'leri hesapla (i. sütunun toplamı - i. sütun i. satırdaki değer)
            false_pos = np.sum(cm[:, i]) - cm[i, i]
            
            # Sıfıra bölünmeyi önle
            if (true_neg + false_pos) > 0:
                spec_i = true_neg / (true_neg + false_pos)
                specificities.append(spec_i)
        
        # Tüm sınıfların specificity değerlerinin ortalamasını al
        specificity = np.mean(specificities) if specificities else 0
    
    return specificity

for key, metric in metrics.items():
    print(f"\n=== {metric['name']} ===")

    k_scores = []
    for k in k_range:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=k, **metric["params"]))
        ])
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        k_scores.append(score)
        print(f"K = {k}: Doğruluk = {score:.4f}")
    
    # K değerlerinin doğruluk skorlarını kaydet
    all_k_scores[metric['name']] = k_scores

    # En iyi K değerini seç
    best_k = k_range[np.argmax(k_scores)]
    best_score = max(k_scores)
    print(f"\n✅ En iyi K değeri ({metric['name']}): {best_k}, Doğruluk: {best_score:.4f}")

    # En iyi modeli tekrar eğit
    best_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k, **metric["params"]))
    ])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Tüm metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred)
    
    # Metrikleri yazdır
    print("\nPerformans Metrikleri:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Sonuçları kaydet
    metrics_results[key] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'best_k': best_k
    }

    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Karmaşıklık Matrisi - {metric['name']}")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek Değer")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{key}.png"
    plt.savefig(cm_path)
    print(f"Karmaşıklık matrisi kaydedildi: {cm_path}")

    # Modeli kaydet
    model_path = f"knn_model_{key}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model dosyası kaydedildi: {model_path}")

    # Örnek tahmin
    sample = X_test.iloc[0:1]
    pred_class = best_model.predict(sample)
    print(f"Örnek Tahmin ({metric['name']}): {le.inverse_transform(pred_class)[0]}")

# K değerlerinin doğruluk skorlarını CSV olarak kaydet
k_scores_df = pd.DataFrame(all_k_scores, index=list(k_range))
k_scores_df.index.name = 'K Değeri'
k_scores_csv_path = "knn_k_scores.csv"
k_scores_df.to_csv(k_scores_csv_path)
print(f"\nK değerlerinin doğruluk skorları CSV olarak kaydedildi: {k_scores_csv_path}")

# K Değerlerinin karşılaştırma grafiği
plt.figure(figsize=(12, 6))
for key, metric in metrics.items():
    plt.plot(k_range, all_k_scores[metric['name']], marker='o', label=metric["name"])

plt.title('K Değerine Göre Doğruluk Skorları')
plt.xlabel('K Değeri')
plt.ylabel('Doğruluk')
plt.xticks(k_range)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("knn_k_values_comparison.png")
plt.show()

# Metrik karşılaştırma tablosu
metrics_df = pd.DataFrame({
    key: {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1'],
        'Specificity': results['specificity'],
        'Best K': results['best_k']
    }
    for key, results in metrics_results.items()
})

print("\n=== Tüm Mesafe Metriklerinin Karşılaştırması ===")
print(metrics_df.round(4).T)

# Metrik karşılaştırma grafiği (Best K hariç)
metrics_plot_df = metrics_df.drop('Best K', axis=0)
plt.figure(figsize=(14, 8))
metrics_plot_df.plot(kind='bar', figsize=(14, 8))
plt.title('Mesafe Metriklerine Göre Performans Karşılaştırması')
plt.xlabel('Performans Metriği')
plt.ylabel('Skor')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Mesafe Metriği')
plt.tight_layout()
plt.savefig("knn_metrics_comparison.png")
plt.show()

# Metrik karşılaştırma tablosunu CSV olarak kaydet
csv_path = "knn_metrics_comparison.csv"
metrics_df.round(4).T.to_csv(csv_path)
print(f"\nMetrik karşılaştırma tablosu CSV olarak kaydedildi: {csv_path}")

# Daha detaylı bir tablo oluştur
detailed_results = []
for key, metric_info in metrics.items():
    result = metrics_results[key]
    detailed_results.append({
        'Mesafe Metriği': metric_info['name'],
        'En İyi K': result['best_k'],
        'Accuracy': result['accuracy'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'F1 Score': result['f1'],
        'Specificity': result['specificity']
    })

# DataFrame oluştur ve CSV olarak kaydet
detailed_df = pd.DataFrame(detailed_results)
detailed_csv_path = "knn_detailed_comparison.csv"
detailed_df.to_csv(detailed_csv_path, index=False)
print(f"\nDetaylı metrik karşılaştırma tablosu CSV olarak kaydedildi: {detailed_csv_path}")