# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_clean_data(file_path, sheet_name=0):
    """
    Load dataset, parse time, clean missing values.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").set_index("Time")

    # Replace commas with NaN and enforce numeric types
    df = df.replace(",", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def select_and_interpolate(df):
    """
    Select relevant sensors and apply time-aware interpolation.
    """
    temp_cols = [c for c in df.columns if "mean_Temp" in c]
    humidity_cols = [c for c in df.columns if "mean_Humidity" in c]
    rssi_cols = [c for c in df.columns if "mean_RSSI" in c]

    target_col = "Plant height (cm)"

    df = df[temp_cols + humidity_cols + rssi_cols + [target_col]]

    # Time-based interpolation
    df = df.interpolate(method="time").dropna()

    return df


def resample_and_engineer_features(df, lags=4):
    """
    Resample data and create lagged features.
    """
    # Enforce hourly sampling
    df = df.resample("1H").mean()

    # Aggregate sensors
    df["Temp_mean"] = df.filter(like="mean_Temp").mean(axis=1)
    df["Humidity_mean"] = df.filter(like="mean_Humidity").mean(axis=1)
    df["RSSI_mean"] = df.filter(like="mean_RSSI").mean(axis=1)

    df = df[["Temp_mean", "Humidity_mean", "RSSI_mean", "Plant height (cm)"]]

    # Lagged features
    for lag in range(1, lags + 1):
        df[f"Temp_lag_{lag}"] = df["Temp_mean"].shift(lag)
        df[f"Humidity_lag_{lag}"] = df["Humidity_mean"].shift(lag)
        df[f"RSSI_lag_{lag}"] = df["RSSI_mean"].shift(lag)

    return df.dropna()


def split_and_scale(df, train_ratio=0.7, val_ratio=0.15):
    """
    Chronological split and normalization.
    """
    target = "Plant height (cm)"
    features = df.columns.drop(target)

    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train.loc[:, features] = scaler_X.fit_transform(train[features])
    val.loc[:, features] = scaler_X.transform(val[features])
    test.loc[:, features] = scaler_X.transform(test[features])

    train.loc[:, target] = scaler_y.fit_transform(train[[target]])
    val.loc[:, target] = scaler_y.transform(val[[target]])
    test.loc[:, target] = scaler_y.transform(test[[target]])

    return train, val, test, scaler_X, scaler_y
