import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX

config = {
    "model_type": "arima",        # "arima", "ar", "sarima"
    "periods": 48,                # her ID için üretilecek veri uzunluğu
    "allow_negative": False,      # Negatif değerlere izin verilsin mi?
    "scale_type": "sum",         # "mean", "sum", "var"
    "arima_order": (1, 0, 0),     # ARIMA için (p,d,q)
    "sarima_order": (1, 0, 0, 12) # SARIMA için (p,d,q,s)
}

def generate_ts(df, id_col, value_col, config):
    out_list = []
    for uid in df[id_col].unique():
        val = df.loc[df[id_col] == uid, value_col].values[0]

        # Model seçimi
        if config["model_type"] == "arima":
            model = ARIMA(np.random.rand(config["periods"]), order=config["arima_order"])
        elif config["model_type"] == "ar":
            model = AutoReg(np.random.rand(config["periods"]), lags=1)
        elif config["model_type"] == "sarima":
            model = SARIMAX(np.random.rand(config["periods"]), order=config["arima_order"], seasonal_order=config["sarima_order"])
        else:
            raise ValueError("Geçersiz model tipi")

        fitted = model.fit()
        sim_data = fitted.predict(start=0, end=config["periods"] - 1)

        # Ölçekleme
        if config["scale_type"] == "mean":
            sim_data *= val / np.mean(sim_data)
        elif config["scale_type"] == "sum":
            sim_data *= val / np.sum(sim_data)
        elif config["scale_type"] == "var":
            sim_data *= np.sqrt(val / np.var(sim_data))

        # Negatifleri kes
        if not config["allow_negative"]:
            sim_data = np.clip(sim_data, 0, None)

        temp_df = pd.DataFrame({
            id_col: uid,
            "t": range(config["periods"]),
            "value": sim_data
        })
        out_list.append(temp_df)

    return pd.concat(out_list, ignore_index=True)




ts= generate_ts(df, id_col="id", value_col="Monthly_Fee", config=config)
print(ts.head())