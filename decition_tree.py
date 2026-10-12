""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report)
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FILE_CSV = "POWER_Point_Daily_20230101_20260822_001d94S_103d02E_LST (1).csv"

df = pd.read_csv(FILE_CSV, sep=";")

# Kolom terakhir (tanpa nama) berisi label; satu kolom kosong lain dibuang
df = df.rename(columns={df.columns[-1]: "KELAS_RADIASI"})
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

df["TANGGAL"] = pd.to_datetime(df["YEAR"], format="%d/%m/%Y")
df = df.drop(columns=["YEAR", "DOY"]).sort_values("TANGGAL").reset_index(drop=True)

df = df.rename(columns={
    "ALLSKY_SFC_SW_DWN": "RADIASI_MATAHARI",
    "T2M": "SUHU",
    "T2MDEW": "TITIK_EMBUN",
    "T2MWET": "SUHU_BOLA_BASAH",
    "QV2M": "KELEMBABAN_SPESIFIK",
    "RH2M": "KELEMBABAN_RELATIF",
    "PS": "TEKANAN_UDARA",
    "WS2M": "KECEPATAN_ANGIN",
    "GWETTOP": "KELEMBABAN_TANAH",
    "PRECTOTCORR": "CURAH_HUJAN",
})
df = df[["TANGGAL", "RADIASI_MATAHARI", "SUHU", "TITIK_EMBUN", "SUHU_BOLA_BASAH",
         "KELEMBABAN_SPESIFIK", "KELEMBABAN_RELATIF", "TEKANAN_UDARA",
         "KECEPATAN_ANGIN", "KELEMBABAN_TANAH", "CURAH_HUJAN", "KELAS_RADIASI"]]

df.to_csv("data_bersih.csv", index=False)
print("Ukuran data:", df.shape)
print("Data kosong per kolom:\n", df.isna().sum()[df.isna().sum() > 0], "\n")
print("Sebaran kelas:\n", df["KELAS_RADIASI"].value_counts(), "\n")

fitur = ["SUHU", "TITIK_EMBUN", "SUHU_BOLA_BASAH", "KELEMBABAN_SPESIFIK",
         "KELEMBABAN_RELATIF", "TEKANAN_UDARA", "KECEPATAN_ANGIN",
         "KELEMBABAN_TANAH", "CURAH_HUJAN"]
urutan_kelas = ["Rendah", "Sedang", "Tinggi"]

X = df[fitur]
y = df["KELAS_RADIASI"]


batas = int(len(df) * 0.8)
X_train, X_test = X.iloc[:batas], X.iloc[batas:]
y_train, y_test = y.iloc[:batas], y.iloc[batas:]
print(f"Latih: {df['TANGGAL'].iloc[0].date()} s/d {df['TANGGAL'].iloc[batas-1].date()} ({len(X_train)} hari)")
print(f"Uji  : {df['TANGGAL'].iloc[batas].date()} s/d {df['TANGGAL'].iloc[-1].date()} ({len(X_test)} hari)\n")


model = make_pipeline(
    SimpleImputer(strategy="median"),         
    DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=20,
        class_weight="balanced",             
        random_state=42,
    ),
)
model.fit(X_train, y_train)
pohon = model.named_steps["decisiontreeclassifier"]


y_pred = model.predict(X_test)
print("Akurasi latih:", round(model.score(X_train, y_train), 3))
print("Akurasi uji  :", round(model.score(X_test, y_test), 3), "\n")
print(classification_report(y_test, y_pred, labels=urutan_kelas, zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=urutan_kelas, cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (data uji)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()


print("Tingkat kepentingan fitur:")
print(pd.Series(pohon.feature_importances_, index=fitur)
        .sort_values(ascending=False).round(3).to_string(), "\n")
print(export_text(pohon, feature_names=fitur))



plt.figure(figsize=(22, 10))
plot_tree(pohon, feature_names=fitur, class_names=list(pohon.classes_),
          filled=True, rounded=True, fontsize=9, proportion=False)
plt.title("Decision Tree - Kelas Radiasi Matahari Jayapura")
plt.tight_layout()
plt.savefig("pohon yang didapat.png", dpi=150)
plt.close()
print("Selesai: data_bersih.csv, decision_tree.png, confusion_matrix.png")