import numpy as np
import pandas as pd
from scipy.stats import beta, gamma, poisson, norm
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# --- Parametreler ---
churn_ratio = 0.05
n = 500
random_state = 42
np.random.seed(random_state)

# --- Beta parametreleri ---
a = 2
b = a * (1 - churn_ratio) / churn_ratio  # b ≈ 38

# target_corr = np.array([
#     [1.0,  0.7, -0.5, -0.3,  0.5],
#     [0.7,  1.0,  0.5,  0.0,  0.0],
#     [-0.5, 0.5,  1.0,  0.8,  0.1],
#     [-0.3, 0.0,  0.8,  1.0, -0.2],
#     [0.5,  0.0,  0.1, -0.2,  1.0]
# ])
#

target_corr = np.array([
    [1.0,   0.2,  -0.15, -0.03,  0.05],
    [0.2,   1.0,   0.2,   0.0,   0.0 ],
    [-0.15, 0.2,   1.0,   0.8,   0.1 ],
    [-0.03, 0.0,   0.8,   1.0,  -0.2 ],
    [0.05,  0.0,   0.1,  -0.2,   1.0 ]
])


variable_specs = {
    'Churn_Score': {
        'dist': 'beta',
        'params': {'a': a, 'b': b},
        'min': 0,
        'max': 1
    },
    'Monthly_Fee': {
        'dist': 'gamma',
        'params': {'a': 2, 'scale': 20},
        'min': 5,
        'max': 200
    },
    'Num_Transactions': {
        'dist': 'poisson',
        'params': {'mu': 5},  # örneğin ortalama 15
        'min': 0,
        'max': 39  # opsiyonel, ancak ppf bunu geçerse kısarsın
    },

    'logins_app_1m_cnt': {
        'dist': 'poisson',
        'params': {'mu': 8},  # Ortalama 8 giriş
        'min': 0,
        'max': 20
    },
    'contacts_call_3m_cnt': {
        'dist': 'poisson',
        'params': {'mu': 12},  # Ortalama 12 çağrı
        'min': 0,
        'max': 40
    }
}

# --- Copula veri üretici ---
def generate_copula_data(specs, corr_matrix, n=500, random_state=42):
    np.random.seed(random_state)
    dim = len(specs)
    mean = np.zeros(dim)
    z = np.random.multivariate_normal(mean, corr_matrix, size=n)
    u = norm.cdf(z)
    data = {}
    for i, (var, info) in enumerate(specs.items()):
        dist = info['dist']
        params = info['params']
        u_col = u[:, i]
        if dist == 'beta':
            raw = beta.ppf(u_col, **params)
        elif dist == 'gamma':
            raw = gamma.ppf(u_col, **params)
        elif dist == 'normal':
            raw = norm.ppf(u_col, **params)
        elif dist == 'poisson':
            raw = poisson.ppf(u_col, **params)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        # Ölçekleme
        scaled = (raw - raw.min()) / (raw.max() - raw.min())
        final = info['min'] + scaled * (info['max'] - info['min'])
        if dist == 'poisson':
            final = np.round(final).astype(int)
        data[var] = final
    return pd.DataFrame(data)

# --- Veri üret ---
df = generate_copula_data(variable_specs, target_corr, n=n)

# --- Dinamik eşik ile churn oluştur ---
target_ratio = churn_ratio
cutoff = np.quantile(df['Churn_Score'], 1 - target_ratio)
df['Churn'] = (df['Churn_Score'] > cutoff).astype(int)

# --- Sonuçlar ---
print("Gerçekleşen churn oranı:", df['Churn'].mean())
print(df.corr())
sns.heatmap(df.iloc[:, 1:].corr(), cmap='coolwarm', annot=True)

print(df.head())

# --- Histogram ---
plt.hist(df['Churn_Score'], bins=30, edgecolor='black')
plt.axvline(cutoff, color='red', linestyle='--', label=f'Cutoff ({cutoff:.3f})')
plt.title("Beta Dağılımından Churn Score")
plt.xlabel("Churn_Score")
plt.ylabel("Frekans")
plt.legend()
plt.show()



###########

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- Özellik ve hedef ayrımı ---
X = df.drop(columns=['Churn', 'Churn_Score'])
y = df['Churn']

# --- Train-test bölme ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Model eğitimi ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- ROC AUC hesapla ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {auc:.4f}")

# --- ROC eğrisi çiz ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
