import time
import numpy as np
import pandas as pd
import gc
from tensorflow.keras import backend as K
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error , root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras.optimizers import Adam


# ------------------------------
# Helper: reshape data for RNNs
# ------------------------------

def reshape_for_rnn(X, timesteps):
    # Ensure X is converted to a 3D array: (samples, timesteps, features)
    n_samples = X.shape[0]
    n_features = X.shape[1] // timesteps
    return X.values.reshape((n_samples, timesteps, n_features))
# ------------------------------
# Model factory
# ------------------------------


def build_model(model_name, timesteps, n_features):
    if model_name == "SVR":
        return SVR(kernel="rbf", C=10, epsilon=0.01)
    
    model = Sequential()
    # Explicitly define the input shape using the Input layer
    model.add(Input(shape=(timesteps, n_features)))
    
    if model_name == "RNN":
        model.add(SimpleRNN(32))
    elif model_name == "LSTM":
        model.add(LSTM(32))
    elif model_name == "GRU":
        model.add(GRU(32))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


# ------------------------------
# Unified training & benchmarking
# ------------------------------
def train_and_benchmark(train_df, val_df, target_col="Plant height (cm)", epochs=30, batch_size=32):
    models = ["SVR", "RNN", "LSTM", "GRU"]
    results = []
    
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values.ravel()
    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values.ravel()

    # Define timesteps for the RNN input shape
    timesteps = 1 
    n_features = X_train.shape[1]
    
    # Reshape for RNN models (samples, timesteps, features)
    X_train_rnn = X_train.reshape((X_train.shape[0], timesteps, n_features))
    X_val_rnn = X_val.reshape((X_val.shape[0], timesteps, n_features))

    for model_name in models:
        print(f"Training {model_name}...")
        
        # FIX: Pass all 3 required arguments to build_model
        model = build_model(model_name, timesteps, n_features)

        start = time.time()
        if model_name == "SVR":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train_rnn, y_train, validation_data=(X_val_rnn, y_val),
                      epochs=epochs, batch_size=batch_size, verbose=0)
        train_time = time.time() - start

        start = time.time()
        if model_name == "SVR":
            y_pred = model.predict(X_val)
        else:
            y_pred = model.predict(X_val_rnn, verbose=0).flatten()
        infer_time = time.time() - start

        # FIX: Use 2026-compliant RMSE function
        rmse = root_mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        results.append([model_name, rmse, mae, train_time, infer_time])

    return pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "Training Time (s)", "Inference Time (s)"])



def walk_forward_validation(data, target_col, timesteps=1, epochs=30, batch_size=32, n_folds=5):
    fold_size = len(data) // (n_folds + 1)
    models = ["SVR", "RNN", "LSTM", "GRU"]
    final_results = []

    for model_name in models:
        rmse_scores = []
        mae_scores = []
        print(f"Evaluating {model_name} across folds...")

        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            val_end = fold_size * (fold + 2)
            
            train_df = data.iloc[:train_end]
            val_df = data.iloc[train_end:val_end]
            
            X_train_raw = train_df.drop(columns=[target_col])
            y_train = train_df[target_col].values.ravel()
            X_val_raw = val_df.drop(columns=[target_col])
            y_val = val_df[target_col].values.ravel()
            
            n_features = X_train_raw.shape[1] // timesteps
            model = build_model(model_name, timesteps, n_features)

            if model_name == "SVR":
                model.fit(X_train_raw.values, y_train)
                y_pred = model.predict(X_val_raw.values)
            else:
                X_train_rnn = reshape_for_rnn(X_train_raw, timesteps)
                X_val_rnn = reshape_for_rnn(X_val_raw, timesteps)
                model.fit(X_train_rnn, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                y_pred = model.predict(X_val_rnn, verbose=0).flatten()
                K.clear_session()

            rmse_scores.append(root_mean_squared_error(y_val, y_pred))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            gc.collect()

        final_results.append([
            model_name, np.mean(rmse_scores), np.std(rmse_scores), 
            np.mean(mae_scores), np.std(mae_scores)
        ])
    
    return pd.DataFrame(final_results, columns=["Model", "RMSE Mean", "RMSE Std", "MAE Mean", "MAE Std"])
