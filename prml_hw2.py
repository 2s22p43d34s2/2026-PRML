import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

SEED_TRAIN = 42
SEED_TEST = 2026
NOISE = 0.20
TRAIN_SAMPLES_PER_CLASS = 500
TEST_SAMPLES_PER_CLASS = 250

OUTPUT_DIR = Path("prml_hw2_outputs_final")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for folder in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.unicode_minus"] = False

COLOR_C0 = "#3b82f6"
COLOR_C1 = "#ef4444"


def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state)

    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X0 = np.column_stack([x, y, z])
    X1 = np.column_stack([-x, y - 1, -z])

    X = np.vstack([X0, X1])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += rng.normal(loc=0.0, scale=noise, size=X.shape)

    return X, labels.astype(int)


def plot_3d_scatter(X, y, title, save_path):
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    colors = np.where(y == 0, COLOR_C0, COLOR_C1)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=18, alpha=0.82, edgecolors="none")

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='C0', markerfacecolor=COLOR_C0, markersize=8),
        Line2D([0], [0], marker='o', color='w', label='C1', markerfacecolor=COLOR_C1, markersize=8),
    ]
    ax.legend(handles=legend_items, title="Classes", loc="upper right", frameon=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, pad=14)
    ax.view_init(elev=24, azim=-58)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_2d_projections(X, y, title_prefix, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))
    colors = np.where(y == 0, COLOR_C0, COLOR_C1)

    settings = [
        (0, 1, "X", "Y", "XY projection"),
        (0, 2, "X", "Z", "XZ projection"),
        (1, 2, "Y", "Z", "YZ projection"),
    ]

    for ax, (i, j, xlabel, ylabel, subtitle) in zip(axes, settings):
        ax.scatter(X[:, i], X[:, j], c=colors, s=14, alpha=0.78, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='C0', markerfacecolor=COLOR_C0, markersize=8),
        Line2D([0], [0], marker='o', color='w', label='C1', markerfacecolor=COLOR_C1, markersize=8),
    ]
    fig.legend(handles=legend_items, title="Classes", loc="upper right", frameon=True)
    fig.suptitle(title_prefix, y=1.02, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_test_metric_bar(df, save_path):
    metrics = ["测试集准确率", "测试集精确率", "测试集召回率", "测试集F1分数"]
    labels = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
    model_names = df["模型"].tolist()

    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * width, df[metric], width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of model performance on the test set")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(confusion_dict, save_path):
    names = list(confusion_dict.keys())
    cols = 3
    rows = int(np.ceil(len(names) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14.5, 8.8))
    axes = np.array(axes).reshape(rows, cols)

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cm = confusion_dict[name]

        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["C0", "C1"])
        ax.set_yticklabels(["C0", "C1"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    total_axes = rows * cols
    for idx in range(len(names), total_axes):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r, c])

    fig.suptitle("Confusion matrices on the test set", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def metric_dict(y_true, y_pred):
    return {
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred, zero_division=0),
        "召回率": recall_score(y_true, y_pred, zero_division=0),
        "F1分数": f1_score(y_true, y_pred, zero_division=0),
    }


X_train, y_train = make_moons_3d(
    n_samples=TRAIN_SAMPLES_PER_CLASS,
    noise=NOISE,
    random_state=SEED_TRAIN
)
X_test, y_test = make_moons_3d(
    n_samples=TEST_SAMPLES_PER_CLASS,
    noise=NOISE,
    random_state=SEED_TEST
)

plot_3d_scatter(
    X_train, y_train,
    "Training set in the 3D feature space",
    FIG_DIR / "图1_训练集三维分布图.png"
)
plot_3d_scatter(
    X_test, y_test,
    "Test set in the 3D feature space",
    FIG_DIR / "图2_测试集三维分布图.png"
)
plot_2d_projections(
    X_train, y_train,
    "Training set projections on coordinate planes",
    FIG_DIR / "图3_训练集二维投影图.png"
)
plot_2d_projections(
    X_test, y_test,
    "Test set projections on coordinate planes",
    FIG_DIR / "图4_测试集二维投影图.png"
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=7,
        min_samples_leaf=2,
        criterion="gini",
        random_state=SEED_TRAIN
    ),
    "AdaBoost + Decision Trees": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED_TRAIN),
        n_estimators=200,
        learning_rate=1.0,
        random_state=SEED_TRAIN
    ),
    "SVM (Linear Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=1.0))
    ]),
    "SVM (Polynomial Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", C=10.0, degree=3, gamma="scale", coef0=1.0))
    ]),
    "SVM (RBF Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale"))
    ]),
    "SVM (Sigmoid Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="sigmoid", C=1.0, gamma="scale", coef0=0.0))
    ]),
}

train_rows = []
test_rows = []
test_confusion_dict = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = metric_dict(y_train, y_train_pred)
    test_metrics = metric_dict(y_test, y_test_pred)

    train_rows.append({
        "模型": model_name,
        "准确率": train_metrics["准确率"],
        "精确率": train_metrics["精确率"],
        "召回率": train_metrics["召回率"],
        "F1分数": train_metrics["F1分数"],
    })
    test_rows.append({
        "模型": model_name,
        "准确率": test_metrics["准确率"],
        "精确率": test_metrics["精确率"],
        "召回率": test_metrics["召回率"],
        "F1分数": test_metrics["F1分数"],
    })

    test_confusion_dict[model_name] = confusion_matrix(y_test, y_test_pred)

order = [
    "AdaBoost + Decision Trees",
    "Decision Tree",
    "SVM (Polynomial Kernel)",
    "SVM (RBF Kernel)",
    "SVM (Linear Kernel)",
    "SVM (Sigmoid Kernel)",
]

train_df = pd.DataFrame(train_rows)
train_df["排序"] = train_df["模型"].map({name: i for i, name in enumerate(order)})
train_df = train_df.sort_values(by="排序").drop(columns="排序").reset_index(drop=True)

test_df = pd.DataFrame(test_rows)
test_df["排序"] = test_df["模型"].map({name: i for i, name in enumerate(order)})
test_df = test_df.sort_values(by="排序").drop(columns="排序").reset_index(drop=True)

plot_test_metric_bar(
    test_df.rename(columns={
        "准确率": "测试集准确率",
        "精确率": "测试集精确率",
        "召回率": "测试集召回率",
        "F1分数": "测试集F1分数",
    }),
    FIG_DIR / "图5_测试集性能对比直方图.png"
)
plot_confusion_matrices(
    test_confusion_dict,
    FIG_DIR / "图6_测试集混淆矩阵.png"
)

train_df.to_csv(TABLE_DIR / "表1_各模型在训练集上的分类性能比较.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(TABLE_DIR / "表2_各模型在测试集上的分类性能比较.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(TABLE_DIR / "实验结果汇总.xlsx", engine="openpyxl") as writer:
    train_df.to_excel(writer, sheet_name="训练集性能", index=False)
    test_df.to_excel(writer, sheet_name="测试集性能", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n表1：各模型在训练集上的分类性能比较")
print(train_df.round(4).to_string(index=False))
print("\n表2：各模型在测试集上的分类性能比较")
print(test_df.round(4).to_string(index=False))
print(f"\n输出目录：{OUTPUT_DIR.resolve()}")
