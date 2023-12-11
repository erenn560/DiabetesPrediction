import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# 1.Aşama: Tanımlamalar
data = pd.read_csv("diabetes.csv")
data.head()

# 2.Aşama: Örnek Çizim
seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]

# İnsülin ve Glikoz oranına göre çizim 

# Renkler ve alpha değerleri
healthy_color = "blue"
diabetic_color = "green"
point_alpha = 0.4

# Sağlıklı insanlar
plt.scatter(saglikli_insanlar.Insulin, saglikli_insanlar.Glucose, c=healthy_color, label="Sağlıklı", alpha=point_alpha)

# Şeker hastası insanlar
plt.scatter(seker_hastalari.Insulin, seker_hastalari.Glucose, c=diabetic_color, label="Şeker Hastası", alpha=point_alpha)

# Eksen etiketleri ve legend
plt.xlabel("İnsülin")
plt.ylabel("Glikoz")
plt.legend()
plt.show()

# 3.Aşama: Normalizasyon İşlemleri
y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"], axis=1)
x = (x_ham_veri - x_ham_veri.min(axis=0)) / (x_ham_veri.max(axis=0) - x_ham_veri.min(axis=0))

# Önce
print("Normalizasyon öncesi ham veriler:\n")
print(x_ham_veri.head())

# Sonra
print("\n\n\nNormalizasyon sonrası veriler:\n")
print(x.head())

# 4.Aşama: Train-Test İşlemleri ve KNN Modeli Oluşturma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("K=5 için Test verileri doğrulama testi sonucu ", knn.score(x_test, y_test))

# 5.Aşama: Doğruluk Oranları
sayac = 1
for k in range(1, 11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train, y_train)
    accuracy = knn_yeni.score(x_test, y_test) * 100
    print(f"{sayac}. Doğruluk oranı: %{accuracy:.2f}")
    sayac += 1

# 6.Aşama: Yeni Hasta Tahmini

# Veri çerçevesini oluştur
x_df = pd.DataFrame(x_ham_veri, columns=x_ham_veri.columns)

# Min-Max ölçekleme
scaler = MinMaxScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x_df), columns=x_df.columns)

# K-En Yakın Komşular modelini oluştur
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_scaled, y)

# Yeni veri noktasını oluştur
new_data = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50], [5, 120, 70, 30, 2, 32.0, 0.300, 25]],
                        columns=x_df.columns)

# Veriyi ölçekle
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)


# Tahminleri al
new_predictions = knn.predict(new_data_scaled)

# İlk tahmini yazdır
print("1. Yeni tahmin :", new_predictions[0])

# İkinci tahmini yazdır
print("2. Yeni tahmin :", new_predictions[1])
