import random
from sklearn.discriminant_analysis import StandardScaler
from datasetPath import PATH
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import chi2_contingency
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim

# Load Dataset
df = pd.read_excel(f"{PATH}BlaBla.xlsx") 

# Pisahkan fitur dan label
X = df.drop(columns=['N', 'UMUR_TAHUN'])  # N adalah label/output dan 'UMUR_TAHUN' sudah di encode dalam kolom A
y = df['N']

# Grouping data X dan y
print("GROUPING VARIABLE".center(75, "="))
print("data variabel".center(75, "="))
print(X)
print("data kelas".center(75, "="))
print(y)
print("=".center(75, "="), "\n")

# Preprocessing
print("Missing value check:")
print(df.isnull().sum())  # Cek missing value

# Optional: Cek outlier dengan boxplot sebelum penanganan
plt.figure(figsize=(10, 6))
sns.boxplot(data=X)
plt.title("Boxplot untuk Deteksi Outlier (Sebelum Penanganan)")
plt.show()

# Fungsi untuk menangani outlier dengan metode IQR (capping/winsorizing)
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Terapkan capping outlier ke semua kolom numerik di X
for col in X.select_dtypes(include=np.number).columns:
    X = cap_outliers_iqr(X, col)

# Cek boxplot setelah penanganan outlier
plt.figure(figsize=(10, 6))
sns.boxplot(data=X)
plt.title("Boxplot untuk Deteksi Outlier (Setelah Penanganan)")
plt.show()

# Encoding umur
def encode_age(age):
    if age <= 20:
        return 1
    elif 21 <= age <= 30:
        return 2
    elif 31 <= age <= 40:
        return 3
    elif 41 <= age <= 50:
        return 4
    else:
        return 5

# Seleksi fitur menggunakan Chi-Squared
selected_features = []

alpha = 0.5  # significance level

for column in X.columns:
    contingency_table = pd.crosstab(df[column], y)
    chi2_val, p, dof, expected = chi2_contingency(contingency_table)
    
    if p < alpha:
        selected_features.append(column)
        print(f"{column}: Significant (p = {p:.4f})")
    else:
        print(f"{column}: Not significant (p = {p:.4f})")

# Filter the original DataFrame to keep only selected features
X_selected = X[selected_features]
print("\nSelected Features: ", selected_features)

# SCENARIO 1 & 2: Under-sampling (RUS) and Over-sampling (SMOTE)
scaler_dict = {}
model_dict = {}

for scenario in range(1, 3):
    print(f"\n{'='*30}\nSCENARIO {scenario}\n{'='*30}")
    if scenario == 1:
        sampler = RandomUnderSampler(random_state=42)
        sampler_name = "RandomUnderSampler"
    else:
        sampler = SMOTE(random_state=42)
        sampler_name = "SMOTE"

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Cek distribusi sebelum dan sesudah sampling
    print(f"Distribusi Sebelum {sampler_name}:\n")
    print(y.value_counts(), "\n")
    plt.figure(figsize=(6,4))
    sns.barplot(x=y.value_counts().index, y=y.value_counts().values, hue=y.value_counts().index, palette='coolwarm', legend=False)
    plt.xticks(ticks=[0,1])
    plt.xlabel("Status")
    plt.ylabel("Jumlah")
    plt.title(f"Distribusi Sebelum {sampler_name}")
    plt.show()

    print(f"\nDistribusi Sesudah {sampler_name}:\n")
    print(pd.Series(y_resampled).value_counts(), "\n")
    plt.figure(figsize=(6,4))
    sns.barplot(x=pd.Series(y_resampled).value_counts().index, 
                y=pd.Series(y_resampled).value_counts().values, 
                hue=pd.Series(y_resampled).value_counts().index, 
                palette='coolwarm', legend=False)
    plt.xticks(ticks=[0,1])
    plt.xlabel("Status")
    plt.ylabel("Jumlah")
    plt.title(f"Distribusi Sesudah {sampler_name}")
    plt.show()

    # Klasifikasi Deep Learning
    # Training dan Testing
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 42)
    print("SPLITTING DATA 20-80".center(75, "="))
    print("instance variabel data training".center(75, "="))
    print(X_train)
    print("instance kelas data training".center(75, "="))
    print(y_train)
    print("instance variabel data testing".center(75, "="))
    print(X_test)
    print("instance kelas data testing".center(75, "="))
    print(y_test)
    print("=".center(75, "="), "\n")

    # === 1. Preprocess Data ===
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32).view(-1, 1)

    # === 2. Define Model ===
    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim):
            super(BinaryClassifier, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = BinaryClassifier(X_train_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === 3. Train ===
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                preds = (outputs > 0.5).float()
                acc = (preds == y_train_tensor).float().mean()
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")

    # === 4. Evaluate ===
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor).numpy()
        y_pred = (y_pred_probs > 0.3).astype("int32")

    print("Akurasi:", accuracy_score(y_test, y_pred))
    print("Presisi:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_probs))

    # Store scaler and model for this scenario
    scaler_dict[scenario] = scaler
    model_dict[scenario] = model

# ==== Input Features (outside the loop, user selects scenario) ====
def get_input_features(scaler):
    print("=".center(60, "="))
    print("         INPUT DATA PASIEN UNTUK KLASIFIKASI         ")
    print("=".center(60, "="))

    # Umur
    while True:
        try:
            A = int(input("Umur Pasien (1-99) = "))
            if 1 <= A <= 99:
                if A < 21: A_k = 1
                elif A < 31: A_k = 2
                elif A < 41: A_k = 3
                elif A < 51: A_k = 4
                else: A_k = 5
                break
            else:
                print("Harap masukkan angka antara 1 sampai 99.")
        except ValueError:
            print("Input tidak valid. Masukkan angka.")

    # Jenis Kelamin
    while True:
        try:
            B_k = int(input("Jenis Kelamin (0=Perempuan, 1=Laki-Laki) = "))
            if B_k in [0, 1]:
                break
            else:
                print("Harap masukkan 0 atau 1.")
        except ValueError:
            print("Input tidak valid.")

    # Gejala-gejala C - M
    symptoms = []
    for gejala in "CDEFGHIJKLM":
        while True:
            res = input(f"Apakah pasien mengalami {gejala}? (Y/N) = ").strip().upper()
            if res in ['Y', 'N']:
                symptoms.append(1 if res == 'Y' else 0)
                break
            else:
                print("Masukkan tidak valid. Harap Y atau N.")

    # Final vector
    feature_list = [A_k, B_k] + symptoms
    test_df = pd.DataFrame([feature_list])

    # Scale the input
    test_scaled = scaler.transform(test_df)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    return test_tensor

# ==== Predict (user selects scenario) ====
while True:
    print("\nPilih data yang digunakan untuk prediksi:")
    print("1. RandomUnderSampler (RUS)")
    print("2. SMOTE")
    try:
        scenario_choice = int(input("Masukkan pilihan (1/2): "))
        if scenario_choice in [1, 2]:
            break
        else:
            print("Pilihan harus 1 atau 2.")
    except ValueError:
        print("Input tidak valid.")

input_tensor = get_input_features(scaler_dict[scenario_choice])
model = model_dict[scenario_choice]
model.eval()
with torch.no_grad():
    pred = model(input_tensor).item()
    print("\n" + "="*60)
    if pred > 0.3:
        print(f"Prediksi: Pasien POSITIF (Probabilitas: {pred:.2f})")
    else:
        print(f"Prediksi: Pasien NEGATIF (Probabilitas: {pred:.2f})")
    print("="*60)