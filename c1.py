import numpy as np
import pandas as pd
from math import ceil, floor

# İlişkili kolon çiftleri (gruplar)
cc_pairs  = [('cc_transaction_all_amt', 'cc_transaction_all_cnt')]
eft_pairs = [('mobile_eft_all_amt', 'mobile_eft_all_cnt')]

def inject_nulls_simple(train_df, test_df,
                        cc_pairs, eft_pairs,
                        frac_cc_null=0.05,       # sadece CC çiftleri NaN atanacak müşteri oranı
                        frac_eft_null=0.05,      # sadece EFT çiftleri NaN atanacak müşteri oranı
                        cut_history_frac=0.03,   # geçmişi kesilecek müşteri oranı
                        min_cut_ratio=0.10,      # kesilecek dönem sayısı: N*min_cut_ratio ..
                        max_cut_ratio=0.30,      # .. N*max_cut_ratio arasında rastgele
                        random_state=42):
    np.random.seed(random_state)

    all_pairs = cc_pairs + eft_pairs  # geçmiş kesmede tüm çiftler kesilecek

    def apply_nulls(df):
        df = df.copy()
        ids = df['cust_id'].unique()
        n = len(ids)

        # 1) CC-only ve EFT-only müşterileri seç (disjoint)
        k_cc  = int(n * frac_cc_null)
        cc_ids = set(np.random.choice(ids, size=k_cc, replace=False))
        remain = np.array([i for i in ids if i not in cc_ids])
        k_eft = int(n * frac_eft_null)
        eft_ids = set(np.random.choice(remain, size=min(k_eft, len(remain)), replace=False))

        # 2) CC-only: CC çiftlerini komple NaN
        for a, c in cc_pairs:
            df.loc[df['cust_id'].isin(cc_ids), [a, c]] = np.nan

        # 3) EFT-only: EFT çiftlerini komple NaN
        for a, c in eft_pairs:
            df.loc[df['cust_id'].isin(eft_ids), [a, c]] = np.nan

        # 4) Geçmiş kesme: bazı müşteriler için başlangıçtan L dönem kes (tüm çiftler)
        k_cut = int(n * cut_history_frac)
        cut_ids = np.random.choice(ids, size=min(k_cut, n), replace=False)

        for cid in cut_ids:
            dates = np.sort(df.loc[df['cust_id'] == cid, 'date'].unique())
            N = len(dates)                 # >=12 varsayımı
            L_min = max(1, ceil(min_cut_ratio * N))
            L_max = min(N-1, floor(max_cut_ratio * N))
            if L_min > L_max:
                continue
            L = np.random.randint(L_min, L_max + 1)  # [L_min, L_max]
            cut_point = dates[L - 1]
            mask = (df['cust_id'] == cid) & (df['date'] <= cut_point)
            for a, c in all_pairs:
                df.loc[mask, [a, c]] = np.nan

        return df

    return apply_nulls(train_df), apply_nulls(test_df)



# CSV'lerini yükle
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# İlişkili kolon çiftleri
cc_pairs  = [('cc_transaction_all_amt', 'cc_transaction_all_cnt')]
eft_pairs = [('mobile_eft_all_amt', 'mobile_eft_all_cnt')]

# Fonksiyonu çağır
train_null, test_null = inject_nulls_simple(
    train_df, test_df,
    cc_pairs=cc_pairs,
    eft_pairs=eft_pairs,
    frac_cc_null=0.05,      # %5 müşteri CC null
    frac_eft_null=0.05,     # %5 müşteri EFT null
    cut_history_frac=0.03,  # %3 müşteri geçmiş kesme
    min_cut_ratio=0.10,     # kesilecek min oran
    max_cut_ratio=0.30,     # kesilecek max oran
    random_state=42
)

#########






# İlişkili kolon çiftleri
paired_cols = [
    ('cc_transaction_all_amt', 'cc_transaction_all_cnt'),
    ('mobile_eft_all_amt', 'mobile_eft_all_cnt')
]

def inject_nulls_with_pairs(train_df, test_df,
                            paired_cols,
                            null_frac=0.05,
                            cut_history_frac=0.03,
                            random_state=42):

    np.random.seed(random_state)
    
    def apply_nulls(df):
        df = df.copy()
        unique_ids = df['cust_id'].unique()
        
        # 1️⃣ Rastgele müşteri alt kümesinde tüm çiftleri NaN yap
        target_ids = np.random.choice(unique_ids,
                                      size=int(len(unique_ids) * null_frac),
                                      replace=False)
        for col1, col2 in paired_cols:
            df.loc[df['cust_id'].isin(target_ids), [col1, col2]] = np.nan

        # 2️⃣ Geçmiş kesme
        cut_ids = np.random.choice(unique_ids,
                                   size=int(len(unique_ids) * cut_history_frac),
                                   replace=False)
        for cid in cut_ids:
            cust_dates = sorted(df.loc[df['cust_id'] == cid, 'date'].unique())
            if len(cust_dates) > 3:
                cut_point = np.random.choice(cust_dates[1:-1])
                for col1, col2 in paired_cols:
                    df.loc[(df['cust_id'] == cid) & (df['date'] <= cut_point),
                           [col1, col2]] = np.nan

        return df

    return apply_nulls(train_df), apply_nulls(test_df)


# Örnek kullanım
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_null, test_null = inject_nulls(train_df, test_df,
                                     null_cols=['kredi_karti', 'eft'],
                                     null_frac=0.05,           # %5 müşteri kolonu NaN
                                     cut_history_frac=0.03)    # %3 müşteri geçmişi kesiliyor

train_null.to_csv("train_with_nulls.csv", index=False)
test_null.to_csv("test_with_nulls.csv", index=False)






import numpy as np
import pandas as pd

def inject_missing_and_cut_history(
    df: pd.DataFrame,
    id_col: str,                 # müşteri id
    date_col: str = None,        # varsa zaman kolonu (örn. 'month')
    target_col: str = None,      # hedef kolon (dokunulmaz)
    amount_count_pairs=None,     # [('cc_amt','cc_cnt'), ('eft_amt','eft_cnt')]
    missing_rate: float = 0.1,   # toplam müşterinin ~%10'unda eksiklik
    cut_rate: float = 0.08,      # toplam müşterinin ~%8'inde geçmiş kesme
    random_state: int = 42
):
    """
    - Eksiklikler rastgele müşterilerde, seçili kolonlarda uygulanır.
    - amount/count tutarlılığı korunur: count=0/null => amount=0/null, amount>0 => count>0.
    - target_col varsa asla değiştirilmez.
    - date_col varsa "geçmiş kesme" yapılır (cut_rate oranında müşteri).
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # Emniyet
    amount_count_pairs = amount_count_pairs or []
    cols_safe = set([id_col] + ([date_col] if date_col else []) + ([target_col] if target_col else []))
    touchable_cols = set(sum(([a,c] for a,c in amount_count_pairs), [])) - cols_safe

    # 1) Müşteri bazında eksik atama
    unique_ids = df[id_col].dropna().unique()
    n_missing_ids = max(1, int(len(unique_ids) * missing_rate))
    ids_for_missing = set(rng.choice(unique_ids, size=n_missing_ids, replace=False))

    # Hangi kolonlara eksik atayalım? (çiftlerin amount tarafını hedefliyoruz; count tutarlılıkla ayarlanır)
    amount_cols = [a for a, _ in amount_count_pairs if a in touchable_cols]

    # Seçili müşterilerde satırların bir kısmına (ör. %40) eksik ata
    row_mask_missing = df[id_col].isin(ids_for_missing) & rng.random(len(df)) < 0.4
    for a, c in amount_count_pairs:
        if a not in df.columns or c not in df.columns: 
            continue
        # amount'ı null yap
        sel = row_mask_missing.copy()
        # target'a dokunma (gereksiz ama güvenlik)
        if target_col and target_col in df.columns:
            sel &= df[target_col].notna() | df[target_col].isna()

        df.loc[sel, a] = np.nan
        # tutarlılık: amount null ise count da null (veya 0). Burada null seçiyoruz.
        df.loc[sel, c] = np.nan

    # 2) Tutarlılık düzeltmeleri (global)
    for a, c in amount_count_pairs:
        if a not in df.columns or c not in df.columns:
            continue
        # count==0 veya count null ise amount 0 veya null olsun (burada null tercih)
        zero_or_null_cnt = (df[c].isna()) | (df[c] == 0)
        df.loc[zero_or_null_cnt, a] = np.nan
        # amount>0 ise count>0 ve non-null
        pos_amt = df[a].fillna(0) > 0
        # count null/0 ise 1 yap (minimal tutarlılık)
        df.loc[pos_amt & (df[c].isna() | (df[c] <= 0)), c] = 1

    # 3) Geçmiş kesme (date_col varsa)
    if date_col and date_col in df.columns:
        # müşteri başına rastgele bir "cut" noktası seç; öncesini null yap
        n_cut_ids = max(1, int(len(unique_ids) * cut_rate))
        ids_for_cut = set(rng.choice(unique_ids, size=n_cut_ids, replace=False))

        # tarih tipini garantiye al
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        for cid in ids_for_cut:
            sub = df[df[id_col] == cid]
            valid_dates = sub[date_col].dropna().unique()
            if len(valid_dates) < 3:
                continue
            cut_point = rng.choice(valid_dates)  # kesme ayı
            mask_before = (df[id_col] == cid) & (df[date_col] < cut_point)

            # geçmişi kes: seçili kolonları null yap (target dahil ETME!)
            for col in touchable_cols:
                if col in df.columns:
                    df.loc[mask_before, col] = np.nan

            # İsteğe bağlı: türetilmiş kolonlar varsa burada da null/yeniden hesaplanır.

    return df






df_out = inject_missing_and_cut_history(
    df=df_in,
    id_col="customer_id",
    date_col="month",                # yoksa None bırak
    target_col="churn",              # yoksa None bırak
    amount_count_pairs=[("cc_amt","cc_cnt"), ("eft_amt","eft_cnt")],
    missing_rate=0.10,               # %10 müşteri için eksiklik
    cut_rate=0.08,                   # %8 müşteri için geçmiş kesme
    random_state=123
)



###########

# eval_missing_impact.py
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ---------- 1) ÖNCE–SONRA: METRİKLER + BOOTSTRAP CI + DeLong ----------
def bootstrap_ci(metric_fn, y_true, y_score, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    stats = []
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.quantile(stats, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

# DeLong p-value (ROC AUC farkı için)
# Kaynak: Sun et al. (2014) & genel kabul görmüş NumPy uyarlaması
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(y_true, y_scores):
    # y_true ∈ {0,1}
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    m, n = len(pos), len(neg)
    tx = _compute_midrank(np.r_[pos, neg])  # midranks
    tx_pos = tx[:m]
    tx_neg = tx[m:]
    auc = (tx_pos.mean() - tx_neg.mean()) / (m + n)
    v01 = (tx_pos - tx_pos.mean()) / (n)
    v10 = (tx_neg - tx_neg.mean()) / (m)
    s01 = v01.var(ddof=1)
    s10 = v10.var(ddof=1)
    auc_var = s01/m + s10/n
    return auc, auc_var

from math import sqrt
from scipy.stats import norm

def delong_test(y_true, s1, s2):
    auc1, var1 = _fast_delong(y_true, s1)
    auc2, var2 = _fast_delong(y_true, s2)
    # Kovaryansı sıfır varsayımı (yaklaşık): konservatif
    var = var1 + var2
    z = (auc1 - auc2) / sqrt(max(var, 1e-12))
    p = 2*(1 - norm.cdf(abs(z)))
    return {"auc1": auc1, "auc2": auc2, "z": z, "p_value": p}

def summarize_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    ll = log_loss(y_true, np.c_[1-y_prob, y_prob])
    auc_ci = bootstrap_ci(roc_auc_score, y_true, y_prob)
    pr_ci  = bootstrap_ci(average_precision_score, y_true, y_prob)
    return {
        "AUC": auc, "AUC_CI": auc_ci,
        "PR_AUC": pr, "PR_AUC_CI": pr_ci,
        "LogLoss": ll
    }

# ---------- 2) SEGMENT BAZLI RAPOR ----------
def segment_report(df, seg_col, y_col, p_cols: dict):
    """
    p_cols = {'before': 'prob_before', 'after': 'prob_after'}
    """
    out = []
    for seg, g in df.groupby(seg_col):
        row = {"segment": seg, "n": len(g)}
        y = g[y_col].values
        for tag, pcol in p_cols.items():
            if pcol in g:
                yp = g[pcol].values
                row[f"AUC_{tag}"] = roc_auc_score(y, yp)
                row[f"PR_{tag}"] = average_precision_score(y, yp)
                row[f"LL_{tag}"] = log_loss(y, np.c_[1-yp, yp])
        if all(k in row for k in ["AUC_before", "AUC_after"]):
            row["dAUC"] = row["AUC_after"] - row["AUC_before"]
            row["dPR"]  = row["PR_after"] - row["PR_before"]
            row["dLL"]  = row["LL_after"] - row["LL_before"]
        out.append(row)
    return pd.DataFrame(out).sort_values("dAUC", ascending=True)

# ---------- 3) MISSING FLAG ETKİSİ ----------
def missing_flag_impact(X, y, model, missing_flag_cols=None, random_state=42):
    """
    - missing_flag_cols: örn. ['is_cc_amt_missing','is_eft_amt_missing',...]
    - Importance: permutation_importance (model agnostik)
    - Basit korelasyon: point-biserial ≈ Pearson(X,y) (ikili y için)
    """
    res = {}
    if missing_flag_cols:
        # korelasyon
        cors = {}
        for c in missing_flag_cols:
            if c in X.columns:
                x = X[c].astype(float).values
                yv = y.astype(float).values
                mx = x.mean(); my = yv.mean()
                num = np.mean((x-mx)*(yv-my))
                den = np.std(x)*np.std(yv) + 1e-12
                cors[c] = float(num/den)
        res["correlation_with_target"] = dict(sorted(cors.items(), key=lambda kv: -abs(kv[1])))

        # permütasyon önem sırası
        pi = permutation_importance(model, X, y, n_repeats=10, random_state=random_state)
        imp = dict(zip(X.columns, pi.importances_mean))
        res["permutation_importance_top"] = sorted(
            {k:v for k,v in imp.items() if k in missing_flag_cols}.items(),
            key=lambda kv: -kv[1]
        )
    return res

# ---------- ÖRNEK AKIŞ (seed sweep ile) ----------
def run_eval(df_before, df_after, y_col, prob_before_col, prob_after_col,
             seg_col=None, seeds=(0,1,2)):
    summary = []
    for seed in seeds:
        # Aynı split ile karşılaştır (örnek: %30 validasyon)
        idx = np.arange(len(df_before))
        rng = np.random.default_rng(seed)
        train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=seed, shuffle=True, stratify=df_before[y_col])
        for tag, dfX, pcol in [("before", df_before, prob_before_col), ("after", df_after, prob_after_col)]:
            metrics = summarize_metrics(dfX[y_col].values[test_idx], dfX[pcol].values[test_idx])
            summary.append({"seed": seed, "phase": tag, **metrics})

        # DeLong testi (test setinde)
        yt = df_before[y_col].values[test_idx]
        s1 = df_before[prob_before_col].values[test_idx]
        s2 = df_after[prob_after_col].values[test_idx]
        dres = delong_test(yt, s1, s2)
        summary.append({"seed": seed, "phase": "delong", **dres})

    summary_df = pd.DataFrame(summary)

    seg_df = None
    if seg_col:
        tmp = pd.DataFrame({
            seg_col: df_before[seg_col],
            y_col: df_before[y_col],
            "prob_before": df_before[prob_before_col],
            "prob_after":  df_after[prob_after_col],
        })
        seg_df = segment_report(tmp, seg_col, y_col, {"before":"prob_before","after":"prob_after"})
    return summary_df, seg_df

# ---- KULLANIM ÖZETİ ----
# 1) df_before ve df_after: Aynı gözlemler (satırlar) için, model çıktı olasılıkları
#    - df_before['prob_before'], df_after['prob_after']
#    - Ortak kolon: y_col (örn. 'target'), opsiyonel: seg_col (örn. 'customer_segment')
# 2) run_eval(...) çağır: summary_df (metrikler + DeLong), seg_df (segment raporu) döner.

