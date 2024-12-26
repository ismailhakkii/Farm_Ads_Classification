import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, make_scorer
from sklearn.decomposition import PCA

# Veri Setinin yolu
file_path = "C:/Users/Dell/PycharmProjects/Farm_Ads_Classification/farm-ads-vect"  # Tam dosya yolu

# Veri Setini Okuma
try:
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            label = int(tokens[0])  # İlk sütun hedef değişken (-1 veya 1)
            features = [0] * 54877  # Özellik vektörü boyutunu önceden belirtiyoruz
            for item in tokens[1:]:
                index, value = map(int, item.split(':'))
                features[index - 1] = value  # Özellik değerlerini dolduruyoruz
            labels.append(label)
            data.append(features)
except FileNotFoundError:
    print(f"Dosya bulunamadı: {file_path}")
    exit()

# Pandas DataFrame'e Dönüştürme
X = pd.DataFrame(data)
y = pd.Series(labels)

# Sınıf Etiketlerini Düzenleme (1 ve 0 olacak şekilde)
y = y.map({1: 1, -1: 0})

# Veriyi Ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Boyut İndirgeme (PCA ile)
pca = PCA(n_components=100)  # Sadece 100 ana bileşen kullan
X_reduced = pca.fit_transform(X_scaled)

# Eğitim ve Test Verisi Ayırma
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Random Forest Modeli için Hiperparametre Arama
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

# RandomizedSearchCV Kullanımı
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=5,  # Daha az kombinasyon deneme
    scoring=make_scorer(recall_score),
    cv=3,
    n_jobs=-1
)

# Modeli Eğitme
random_search.fit(X_train, y_train)

# En İyi Model ve Hiperparametreler
best_model = random_search.best_estimator_
print("En İyi Hiperparametreler:", random_search.best_params_)

# Test Verisinde Tahmin Yapma
y_pred = best_model.predict(X_test)

# Performans Metrikleri
cm = confusion_matrix(y_test, y_pred)
tp = cm[1, 1]
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = accuracy_score(y_test, y_pred)

# Sonuçları Yazdırma
print("Confusion Matrix:")
print(cm)
print(f"Sensitivity (Duyarlılık): {sensitivity:.2f}")
print(f"Specificity (Özgüllük): {specificity:.2f}")
print(f"Accuracy (Doğruluk): {accuracy:.2f}")
