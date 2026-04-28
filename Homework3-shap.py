import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from tensorflow.keras.models import load_model


SHAP_BUNDLE_DIR = "shap_bundle"
SHAP_OUTPUT_DIR = "shap_outputs"
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)


BACKGROUND_SIZE = 100     # 背景样本数，越大越慢
EXPLAIN_SIZE = 50         # 解释测试样本数，越大越慢
NSAMPLES = 200            # SHAP内部采样数
LOCAL_TOPK = 20           # 单样本局部解释展示前20项


def load_bundle():
    """读取 ceshi.py 保存的同源模型与数据。"""
    model_path = os.path.join(SHAP_BUNDLE_DIR, "multivariate_lstm.keras")
    feature_path = os.path.join(SHAP_BUNDLE_DIR, "feature_cols.json")
    config_path = os.path.join(SHAP_BUNDLE_DIR, "config.json")
    x_train_path = os.path.join(SHAP_BUNDLE_DIR, "X_train.npy")
    x_test_path = os.path.join(SHAP_BUNDLE_DIR, "X_test.npy")
    y_test_path = os.path.join(SHAP_BUNDLE_DIR, "y_test_original.npy")
    test_times_path = os.path.join(SHAP_BUNDLE_DIR, "test_times.npy")

    needed_files = [
        model_path, feature_path, config_path,
        x_train_path, x_test_path, y_test_path, test_times_path
    ]
    for f in needed_files:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"缺少文件：{f}\n请先运行 ceshi.py，确保已经生成 shap_bundle 文件夹。"
            )

    model = load_model(model_path)

    with open(feature_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_test_original = np.load(y_test_path)
    test_times = np.load(test_times_path)

    return model, feature_cols, config, X_train, X_test, y_test_original, test_times


def compute_shap_values(model, X_train, X_test, seed):
    """
    使用 GradientExplainer 计算 SHAP 值。
    这里解释的是 ceshi.py 那一轮训练得到的同一个模型。
    """
    bg_size = min(BACKGROUND_SIZE, len(X_train))
    ex_size = min(EXPLAIN_SIZE, len(X_test))

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=bg_size, replace=False)
    ex_idx = rng.choice(len(X_test), size=ex_size, replace=False)

    background = X_train[bg_idx]
    X_explain = X_test[ex_idx]

    print("背景样本 shape:", background.shape)
    print("待解释样本 shape:", X_explain.shape)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_explain, nsamples=NSAMPLES)

    # 单输出回归模型通常会返回 list，取第一个即可
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    # 兼容 shape=(样本数, 时间步, 特征数, 1) 的情况
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    if shap_values.ndim != 3:
        raise ValueError(
            f"SHAP values 维度异常，当前 shape={shap_values.shape}，"
            f"预期应为 (样本数, 时间步, 特征数)"
        )

    np.save(os.path.join(SHAP_OUTPUT_DIR, "shap_values.npy"), shap_values)
    np.save(os.path.join(SHAP_OUTPUT_DIR, "X_explain.npy"), X_explain)

    return shap_values, X_explain


def save_global_feature_importance(shap_values, feature_cols):
    """
    全局特征重要性：
    对 |SHAP| 在“样本维 + 时间维”上取平均
    """
    feature_importance = np.abs(shap_values).mean(axis=(0, 1))

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": feature_importance
    }).sort_values("mean_abs_shap", ascending=False)

    df_imp.to_csv(
        os.path.join(SHAP_OUTPUT_DIR, "global_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["feature"][::-1], df_imp["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_OUTPUT_DIR, "global_feature_importance.png"),
        dpi=300
    )
    plt.close()

    return df_imp


def save_timestep_importance(shap_values, window_size):
    """
    时间步重要性：
    对 |SHAP| 在“样本维 + 特征维”上取平均
    """
    timestep_importance = np.abs(shap_values).mean(axis=(0, 2))
    lag_labels = [f"t-{window_size - i}" for i in range(window_size)]

    df_time = pd.DataFrame({
        "time_step": lag_labels,
        "mean_abs_shap": timestep_importance
    })

    df_time.to_csv(
        os.path.join(SHAP_OUTPUT_DIR, "timestep_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(lag_labels, timestep_importance, marker="o")
    plt.xlabel("Historical time step")
    plt.ylabel("Mean |SHAP value|")
    plt.title("Time-step Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_OUTPUT_DIR, "timestep_importance.png"),
        dpi=300
    )
    plt.close()

    return df_time


def save_feature_time_heatmap(shap_values, feature_cols, window_size):
    """
    热力图：
    每个“时间步-特征”的平均绝对 SHAP 值
    """
    heatmap = np.abs(shap_values).mean(axis=0)  # (时间步, 特征数)
    lag_labels = [f"t-{window_size - i}" for i in range(window_size)]

    np.save(os.path.join(SHAP_OUTPUT_DIR, "feature_time_heatmap.npy"), heatmap)

    plt.figure(figsize=(max(10, len(feature_cols) * 0.8), 6))
    plt.imshow(heatmap, aspect="auto")
    plt.colorbar(label="Mean |SHAP value|")
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.yticks(np.arange(window_size), lag_labels)
    plt.xlabel("Feature")
    plt.ylabel("Historical time step")
    plt.title("Feature-Time SHAP Heatmap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_OUTPUT_DIR, "feature_time_heatmap.png"),
        dpi=300
    )
    plt.close()


def save_local_explanation_bar(shap_values, X_explain, feature_cols, window_size, sample_index=0):
    """
    单样本局部解释：
    选一个测试样本，把 (时间步, 特征数) 展平，
    取绝对值最大的前 LOCAL_TOPK 项画条形图。
    """
    sample_values = shap_values[sample_index]   # (时间步, 特征数)
    sample_data = X_explain[sample_index]       # (时间步, 特征数)

    flat_values = sample_values.reshape(-1)
    flat_data = sample_data.reshape(-1)

    flat_feature_names = []
    for i in range(window_size):
        lag = window_size - i
        for feat in feature_cols:
            flat_feature_names.append(f"{feat}@t-{lag}")

    abs_idx = np.argsort(np.abs(flat_values))[::-1][:LOCAL_TOPK]

    df_local = pd.DataFrame({
        "feature_time": [flat_feature_names[i] for i in abs_idx],
        "shap_value": [flat_values[i] for i in abs_idx],
        "input_value": [flat_data[i] for i in abs_idx],
        "abs_shap": [abs(flat_values[i]) for i in abs_idx]
    }).sort_values("abs_shap", ascending=True)

    df_local.to_csv(
        os.path.join(SHAP_OUTPUT_DIR, f"local_explanation_sample{sample_index}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(df_local["feature_time"], df_local["shap_value"])
    plt.xlabel("SHAP value")
    plt.ylabel("Feature@Time")
    plt.title(f"Local SHAP Explanation (sample {sample_index})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_OUTPUT_DIR, f"local_explanation_sample{sample_index}.png"),
        dpi=300
    )
    plt.close()


def main():
    print("开始读取主实验保存的同源模型与数据...")
    model, feature_cols, config, X_train, X_test, y_test_original, test_times = load_bundle()

    print("模型与数据读取完成")
    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)

    shap_values, X_explain = compute_shap_values(
        model=model,
        X_train=X_train,
        X_test=X_test,
        seed=config["seed"]
    )

 
    df_imp = save_global_feature_importance(shap_values, feature_cols)

    save_timestep_importance(shap_values, config["window_size"])
    save_feature_time_heatmap(shap_values, feature_cols, config["window_size"])
    save_local_explanation_bar(
        shap_values, X_explain, feature_cols, config["window_size"], sample_index=0
    )

if __name__ == "__main__":
    main()