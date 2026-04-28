import os
import random
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


warnings.filterwarnings("ignore")


# =========================
# 1. 基本配置
# =========================
SEED = 42
WINDOW_SIZE = 24          
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
EPOCHS = 30                
BATCH_SIZE = 64

CSV_PATH = r"C:\Users\33871\Desktop\LSTM-Multivariate_pollution.csv"

# 输出目录
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SHAP联动文件保存目录
SHAP_BUNDLE_DIR = "shap_bundle"
os.makedirs(SHAP_BUNDLE_DIR, exist_ok=True)


def set_seed(seed=42):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess(csv_path):
    """
    读取并预处理数据。
    兼容两种格式：

    1. 原始格式：
       No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir

    2. 处理后格式：
       date, pollution, dew, temp, press, wnd_dir, wnd_spd, snow, rain
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("原始列名：", df.columns.tolist())

    # =========================
    # 情况1：原始数据格式
    # =========================
    if "pm2.5" in df.columns:
        df.replace("NA", np.nan, inplace=True)

        numeric_cols = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if "No" in df.columns:
            df.drop(columns=["No"], inplace=True)

        df = df.dropna(subset=["pm2.5"]).copy()

        df["cbwd"] = df["cbwd"].fillna("Unknown")

        other_num_cols = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        df[other_num_cols] = df[other_num_cols].ffill().bfill()

        df = pd.get_dummies(df, columns=["cbwd"], drop_first=False)

        return df

    # =========================
    # 情况2：处理后数据格式
    # =========================
    elif "pollution" in df.columns:
        rename_map = {
            "pollution": "pm2.5",
            "dew": "DEWP",
            "temp": "TEMP",
            "press": "PRES",
            "wnd_spd": "Iws",
            "snow": "Is",
            "rain": "Ir"
        }
        df.rename(columns=rename_map, inplace=True)

        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
            df.drop(columns=["date"], inplace=True)
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        else:
            df["datetime"] = pd.date_range(start="2010-01-01", periods=len(df), freq="H")

        numeric_cols = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["pm2.5"]).copy()

        other_num_cols = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        existing_num_cols = [col for col in other_num_cols if col in df.columns]
        if existing_num_cols:
            df[existing_num_cols] = df[existing_num_cols].ffill().bfill()

        if "wnd_dir" in df.columns:
            df["wnd_dir"] = df["wnd_dir"].fillna("Unknown")
            df = pd.get_dummies(df, columns=["wnd_dir"], drop_first=False)

        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    else:
        raise ValueError(f"无法识别数据列名，当前列为：{df.columns.tolist()}")


def time_split(df, train_ratio=0.7, val_ratio=0.15):
    """按照时间顺序划分训练集、验证集、测试集。"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_sequences(df, feature_cols, target_col, window_size,
                     x_scaler=None, y_scaler=None, fit_scaler=False):
    """
    将表格数据构造成 LSTM 可用的序列样本。
    输入形状: (样本数, 时间步, 特征数)
    输出形状: (样本数,)
    """
    X_raw = df[feature_cols].values
    y_raw = df[[target_col]].values
    times = df["datetime"].values
    y_true_original = df[target_col].values

    if fit_scaler:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X_raw)
        y_scaled = y_scaler.fit_transform(y_raw)
    else:
        X_scaled = x_scaler.transform(X_raw)
        y_scaled = y_scaler.transform(y_raw)

    X_seq, y_seq = [], []
    seq_times, seq_y_true_original = [], []

    for i in range(window_size, len(df)):
        X_seq.append(X_scaled[i - window_size:i])
        y_seq.append(y_scaled[i, 0])
        seq_times.append(times[i])
        seq_y_true_original.append(y_true_original[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    return X_seq, y_seq, np.array(seq_times), np.array(seq_y_true_original), x_scaler, y_scaler


def build_lstm_model(window_size, n_features):
    """构建 LSTM 模型。"""
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def evaluate_model(model, X_test, y_scaler, test_times, y_test_original, model_name):
    """预测并计算评价指标。"""
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_test_original.reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    result_df = pd.DataFrame({
        "datetime": test_times,
        "y_true": y_true,
        "y_pred": y_pred
    })

    metrics = {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return result_df, metrics


def plot_loss(history, model_name):
    """绘制训练损失曲线。"""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_loss.png"), dpi=300)
    plt.close()


def plot_predictions(result_df, model_name, num_points=300):
    """绘制真实值与预测值对比图。"""
    plot_df = result_df.iloc[:num_points].copy()

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["datetime"], plot_df["y_true"], label="True PM2.5")
    plt.plot(plot_df["datetime"], plot_df["y_pred"], label="Predicted PM2.5")
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.title(f"{model_name} Prediction vs True")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_prediction.png"), dpi=300)
    plt.close()


def save_multivariate_artifacts(model, feature_cols, X_train, X_test, test_times, y_test_original):
    """
    保存 SHAP 分析必须使用的同一轮训练结果。
    """
    # 1. 保存模型
    model.save(os.path.join(SHAP_BUNDLE_DIR, "multivariate_lstm.keras"))

    # 2. 保存特征名
    with open(os.path.join(SHAP_BUNDLE_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # 3. 保存配置
    config = {
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "batch_size": BATCH_SIZE,
        "csv_path": CSV_PATH
    }
    with open(os.path.join(SHAP_BUNDLE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 4. 保存序列样本
    np.save(os.path.join(SHAP_BUNDLE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SHAP_BUNDLE_DIR, "X_test.npy"), X_test)

    # 5. 保存测试时间与真实值
    np.save(os.path.join(SHAP_BUNDLE_DIR, "test_times.npy"), test_times)
    np.save(os.path.join(SHAP_BUNDLE_DIR, "y_test_original.npy"), y_test_original)


def run_experiment(df, feature_cols, model_name, save_for_shap=False):
    """完整跑一组实验：划分、缩放、构造序列、训练、评估、画图。"""
    target_col = "pm2.5"

    train_df, val_df, test_df = time_split(df, TRAIN_RATIO, VAL_RATIO)

    # 构造训练集
    X_train, y_train, _, _, x_scaler, y_scaler = create_sequences(
        train_df, feature_cols, target_col, WINDOW_SIZE, fit_scaler=True
    )

    # 构造验证集
    X_val, y_val, _, _, _, _ = create_sequences(
        val_df, feature_cols, target_col, WINDOW_SIZE,
        x_scaler=x_scaler, y_scaler=y_scaler, fit_scaler=False
    )

    # 构造测试集
    X_test, y_test, test_times, y_test_original, _, _ = create_sequences(
        test_df, feature_cols, target_col, WINDOW_SIZE,
        x_scaler=x_scaler, y_scaler=y_scaler, fit_scaler=False
    )

    print(f"\n====== {model_name} 数据形状 ======")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    model = build_lstm_model(WINDOW_SIZE, len(feature_cols))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    plot_loss(history, model_name)

    result_df, metrics = evaluate_model(
        model, X_test, y_scaler, test_times, y_test_original, model_name
    )

    plot_predictions(result_df, model_name)

    # 保存预测结果
    result_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv"), index=False)

    # 如果是多变量模型，额外保存给 SHAP 用的同源文件
    if save_for_shap:
        save_multivariate_artifacts(
            model=model,
            feature_cols=feature_cols,
            X_train=X_train,
            X_test=X_test,
            test_times=test_times,
            y_test_original=y_test_original
        )

    return metrics


def main():
    print("程序开始运行...")
    set_seed(SEED)

    print("开始读取和预处理数据...")
    df = load_and_preprocess(CSV_PATH)
    print("预处理完成")
    print("预处理后数据形状:", df.shape)
    print("字段如下:")
    print(df.columns.tolist())

    # 保存整体 PM2.5 曲线图
    plt.figure(figsize=(12, 5))
    plt.plot(df["datetime"], df["pm2.5"])
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 Time Series")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pm25_time_series.png"), dpi=300)
    plt.close()

    # 单变量特征
    univariate_features = ["pm2.5"]

    # 多变量特征：除时间列外，其余都作为输入
    exclude_cols = ["datetime"]
    multivariate_features = [col for col in df.columns if col not in exclude_cols]

    print("\n单变量特征:", univariate_features)
    print("\n多变量特征:", multivariate_features)

    metrics_list = []

    # 单变量 LSTM
    metrics_uni = run_experiment(
        df, univariate_features, "univariate_lstm", save_for_shap=False
    )
    metrics_list.append(metrics_uni)

    # 多变量 LSTM
    metrics_multi = run_experiment(
        df, multivariate_features, "multivariate_lstm", save_for_shap=True
    )
    metrics_list.append(metrics_multi)

    # 保存指标对比表
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_comparison.csv"), index=False)
    print(metrics_df)

if __name__ == "__main__":
    main()