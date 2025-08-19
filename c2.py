import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from catboost import CatBoostClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

def build_base_models(seed=42):
    models = {
        "cat": CatBoostClassifier(
            random_seed=seed, depth=6, learning_rate=0.05, n_estimators=1000,
            loss_function="Logloss", eval_metric="AUC", verbose=False
        ),
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None, C=1.0, random_state=seed))
        ])
    }
    if _HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=1500, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.03, reg_lambda=1.0, objective="binary:logistic",
            eval_metric="auc", random_state=seed, tree_method="hist"
        )
    return models

def optimize_weights(oof_probs_dict, y_true, grid_steps=11):
    """
    oof_probs_dict: {model_name: np.array[oof_proba]}
    y_true: np.array
    grid_steps: ağırlık 0..1 aralığını 'grid_steps' parçaya böler (örn 11 -> 0.0,0.1,...,1.0)
    """
    names = list(oof_probs_dict.keys())
    P = [oof_probs_dict[n] for n in names]
    best = {"auc": -1, "weights": None}

    # 2 veya 3 model için küçük ızgara taraması (toplamı 1'e normalize ediliyor)
    if len(P) == 1:
        w = np.array([1.0])
        auc = roc_auc_score(y_true, P[0])
        return names, w, auc

    grid = np.linspace(0, 1, grid_steps)
    if len(P) == 2:
        for a in grid:
            w = np.array([a, 1-a])
            blend = w[0]*P[0] + w[1]*P[1]
            auc = roc_auc_score(y_true, blend)
            if auc > best["auc"]:
                best = {"auc": auc, "weights": w}
    elif len(P) == 3:
        for a in grid:
            for b in grid:
                if a + b > 1: 
                    continue
                c = 1 - a - b
                w = np.array([a, b, c])
                blend = w[0]*P[0] + w[1]*P[1] + w[2]*P[2]
                auc = roc_auc_score(y_true, blend)
                if auc > best["auc"]:
                    best = {"auc": auc, "weights": w}
    else:
        # 3+ model varsa basit ortalama fallback (istersen Dirichlet/nelder-mead eklenebilir)
        k = len(P)
        w = np.ones(k) / k
        best = {"auc": roc_auc_score(y_true, sum(P)/k), "weights": w}

    return names, best["weights"], best["auc"]

def train_cv_ensemble(train, features, cv_windows, ratio_churn=0.1, seed=42, grid_steps=11):
    """
    cv_windows: her fold için (val_start, val_end) datetime tuple listesi
                Örn: [(pd.Timestamp('2018-04-01'), pd.Timestamp('2018-04-30')), ...]
    Döner:
      - fitted_models: {model_name: fitted_estimator_on_full_train}
      - best_weights:  {model_name: weight}
      - cv_report:     per-fold ve ortalama AUC/F1
    """
    train = train.sort_values("date").reset_index(drop=True)
    y_all = train["churn"].values

    cv_aucs_val, cv_f1s_val, cv_aucs_tr, cv_f1s_tr = [], [], [], []
    per_fold_weights = []

    for i, (vst, ved) in enumerate(cv_windows, start=1):
        tr_idx = train["date"] < vst
        va_idx = (train["date"] >= vst) & (train["date"] <= ved)

        tr, va = train.loc[tr_idx], train.loc[va_idx]
        X_tr, y_tr = tr[features].values, tr["churn"].values
        X_va, y_va = va[features].values, va["churn"].values

        print(f"\n{'='*40}\nFold {i}: {vst.date()} → {ved.date()} | "
              f"train {tr.shape[0]} / valid {va.shape[0]}")

        # Base modelleri kur ve eğit
        models = build_base_models(seed=seed)
        oof_probs = {}
        tr_probs  = {}

        for name, est in models.items():
            est.fit(X_tr, y_tr)
            oof_probs[name] = est.predict_proba(X_va)[:, 1]
            tr_probs[name]  = est.predict_proba(X_tr)[:, 1]

        # Ağırlıkları validasyonda AUC maksimize ederek bul
        names, w, auc_w = optimize_weights(oof_probs, y_va, grid_steps=grid_steps)
        per_fold_weights.append(dict(zip(names, w)))
        blend_va = sum(w[k]*oof_probs[n] for k, n in enumerate(names))

        # F1 için (opsiyonel) ratio_churn tabanlı eşik
        thr_rank = max(1, int(len(y_va) * ratio_churn))
        thr = np.partition(blend_va, -thr_rank)[-thr_rank]  # üst yüzde eşiği
        y_hat_va = (blend_va >= thr).astype(int)

        # Train tarafında rapor (overfit kontrolü)
        blend_tr = sum(w[k]*tr_probs[n] for k, n in enumerate(names))
        thr_rank_tr = max(1, int(len(y_tr) * ratio_churn))
        thr_tr = np.partition(blend_tr, -thr_rank_tr)[-thr_rank_tr]
        y_hat_tr = (blend_tr >= thr_tr).astype(int)

        auc_va = roc_auc_score(y_va, blend_va)
        auc_tr = roc_auc_score(y_tr, blend_tr)
        from sklearn.metrics import f1_score
        f1_va  = f1_score(y_va, y_hat_va)
        f1_tr  = f1_score(y_tr, y_hat_tr)

        print(f"AUC valid: {auc_va:.4f} | AUC train: {auc_tr:.4f} | "
              f"F1 valid: {f1_va:.4f} | F1 train: {f1_tr:.4f} | "
              f"weights: {dict(zip(names, np.round(w,3)))}")

        cv_aucs_val.append(auc_va); cv_f1s_val.append(f1_va)
        cv_aucs_tr.append(auc_tr);  cv_f1s_tr.append(f1_tr)

    # Fold ağırlıklarını ortala (stabil final ağırlık)
    all_names = list(build_base_models(seed=seed).keys())
    avg_w = {n: 0.0 for n in all_names}
    for d in per_fold_weights:
        for n, v in d.items():
            avg_w[n] += v
    for n in avg_w:
        avg_w[n] = avg_w[n] / len(per_fold_weights) if n in avg_w else 0.0

    # Tüm train üzerinde final modelleri eğit
    fitted_models = build_base_models(seed=seed)
    X_full, y_full = train[features].values, train["churn"].values
    for n, est in fitted_models.items():
        est.fit(X_full, y_full)

    cv_report = {
        "val_auc_mean": float(np.mean(cv_aucs_val)),
        "val_f1_mean":  float(np.mean(cv_f1s_val)),
        "tr_auc_mean":  float(np.mean(cv_aucs_tr)),
        "tr_f1_mean":   float(np.mean(cv_f1s_tr)),
        "per_fold_weights": per_fold_weights,
        "avg_weights": avg_w
    }
    print("\n" + "-"*40)
    print("CV Summary:",
          {k: (np.round(v,4) if isinstance(v, float) else v) for k, v in cv_report.items() if "weights" not in k})
    print("Avg weights:", {k: round(v,3) for k, v in avg_w.items()})

    return fitted_models, avg_w, cv_report






# örnek: ay bazlı pencereler (her biri 1 ay)
def month_window(yyyymm):
    st = pd.to_datetime(yyyymm + "01", format="%Y%m%d")
    ed = (st + pd.offsets.MonthEnd(0))
    return (st, ed)

cv_mm = ["201804","201805","201806","201807","201808","201809","201810",
         "201811","201812","201901","201902","201903","201904","201905","201906"]
cv_windows = [month_window(m) for m in cv_mm]

fitted_models, best_w, cv_rep = train_cv_ensemble(
    train=df_train,                # 'date' (datetime64) ve 'churn' içermeli
    features=feature_list,
    cv_windows=cv_windows,
    ratio_churn=0.10,
    seed=42,
    grid_steps=11  # 0.0..1.0 adım=0.1
)


def predict_ensemble(models_dict, weights_dict, X):
    # normalize (güvenlik)
    wsum = sum(weights_dict.get(k, 0.0) for k in models_dict.keys())
    if wsum == 0:
        wsum = 1.0
    probs = None
    for name, est in models_dict.items():
        w = weights_dict.get(name, 0.0) / wsum
        p = est.predict_proba(X)[:, 1]
        probs = (w*p) if probs is None else (probs + w*p)
    return probs

# --- test setinde çalıştır ---
# df_test: test DataFrame'in (id_col + features)
# Kaggle genelde 'id' ve 'prediction' kolonu ister; yarışmana göre adları ayarla.
id_col = "customer_id"   # senin test id kolonu
sub_col = "prediction"   # istenen çıktı kolon adı (örn. "churn_probability")

X_test = df_test[feature_list].values
test_proba = predict_ensemble(fitted_models, best_w, X_test)

# (Opsiyonel) Eşikli sınıf etiketi de istersen:
ratio_churn = 0.10
k = max(1, int(len(test_proba) * ratio_churn))
thr_test = np.partition(test_proba, -k)[-k]
y_pred_label = (test_proba >= thr_test).astype(int)

# Kaggle submission (olasılık gönder)
submission = pd.DataFrame({
    id_col: df_test[id_col].values,
    sub_col: test_proba
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape", submission.shape)

# Eğer yarışma sınıf etiketi istiyorsa (nadirdir):
# pd.DataFrame({id_col: df_test[id_col].values, "target": y_pred_label}).to_csv("submission_labels.csv", index=False)
