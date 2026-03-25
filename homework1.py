import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#1. 读取数据
file_path = r"C:\Users\33871\Desktop\Data4Regression.xlsx"
train_df = pd.read_excel(file_path, sheet_name=0)
test_df = pd.read_excel(file_path, sheet_name=1)
x_train = train_df.iloc[:, 0].values.reshape(-1, 1)
y_train = train_df.iloc[:, 1].values.reshape(-1, 1)
x_test = test_df.iloc[:, 0].values.reshape(-1, 1)
y_test = test_df.iloc[:, 1].values.reshape(-1, 1)
# 2. 工具函数
def evaluate(y_true, y_pred):
    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

def add_bias(x):
    return np.hstack([np.ones((x.shape[0], 1)), x])

def least_squares(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def gradient_descent(X, y, lr=0.01, epochs=10000):
    n, d = X.shape
    w = np.zeros((d, 1))
    for _ in range(epochs):
        y_pred = X @ w
        grad = (2 / n) * X.T @ (y_pred - y)
        w -= lr * grad
    return w

def newton_method(X, y, epochs=20):
    n, d = X.shape
    w = np.zeros((d, 1))
    H = (2 / n) * (X.T @ X)
    H_inv = np.linalg.pinv(H)
    for _ in range(epochs):
        y_pred = X @ w
        grad = (2 / n) * X.T @ (y_pred - y)
        w -= H_inv @ grad
    return w

def minmax_scale_by_train(x_train, x_test):
    x_min = x_train.min()
    x_max = x_train.max()
    x_train_scaled = 2 * (x_train - x_min) / (x_max - x_min) - 1
    x_test_scaled = 2 * (x_test - x_min) / (x_max - x_min) - 1
    return x_train_scaled, x_test_scaled, x_min, x_max

def polynomial_features(x, degree):
    X = np.ones((x.shape[0], 1))
    for d in range(1, degree + 1):
        X = np.hstack([X, x ** d])
    return X

def gaussian_features(x, centers, sigma):
    X = np.ones((x.shape[0], 1))
    for c in centers:
        phi = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        X = np.hstack([X, phi])
    return X

# 3. 线性回归三种方法

X_train_lin = add_bias(x_train)
X_test_lin = add_bias(x_test)

w_ls = least_squares(X_train_lin, y_train)
w_gd = gradient_descent(X_train_lin, y_train, lr=0.01, epochs=10000)
w_nt = newton_method(X_train_lin, y_train, epochs=20)

y_train_ls = X_train_lin @ w_ls
y_test_ls = X_test_lin @ w_ls
y_train_gd = X_train_lin @ w_gd
y_test_gd = X_test_lin @ w_gd
y_train_nt = X_train_lin @ w_nt
y_test_nt = X_test_lin @ w_nt

table1 = pd.DataFrame({
    "方法": ["最小二乘法", "梯度下降法", "牛顿法"],
    "w0": [w_ls[0, 0], w_gd[0, 0], w_nt[0, 0]],
    "w1": [w_ls[1, 0], w_gd[1, 0], w_nt[1, 0]]
})

table2 = pd.DataFrame({
    "方法": ["最小二乘法", "梯度下降法", "牛顿法"],
    "训练MSE": [evaluate(y_train, y_train_ls)[0], evaluate(y_train, y_train_gd)[0], evaluate(y_train, y_train_nt)[0]],
    "测试MSE": [evaluate(y_test, y_test_ls)[0], evaluate(y_test, y_test_gd)[0], evaluate(y_test, y_test_nt)[0]],
    "训练R2": [evaluate(y_train, y_train_ls)[1], evaluate(y_train, y_train_gd)[1], evaluate(y_train, y_train_nt)[1]],
    "测试R2": [evaluate(y_test, y_test_ls)[1], evaluate(y_test, y_test_gd)[1], evaluate(y_test, y_test_nt)[1]]
})

# 4. 多项式回归（2~13阶）

x_train_poly, x_test_poly, x_min_poly, x_max_poly = minmax_scale_by_train(x_train, x_test)

poly_results = []
poly_models = {}

for degree in range(2, 14):
    X_train_p = polynomial_features(x_train_poly, degree)
    X_test_p = polynomial_features(x_test_poly, degree)

    w_poly = least_squares(X_train_p, y_train)

    y_train_pred = X_train_p @ w_poly
    y_test_pred = X_test_p @ w_poly

    train_mse, train_r2 = evaluate(y_train, y_train_pred)
    test_mse, test_r2 = evaluate(y_test, y_test_pred)

    poly_results.append({
        "阶数": degree,
        "训练MSE": train_mse,
        "测试MSE": test_mse,
        "训练R2": train_r2,
        "测试R2": test_r2
    })

    poly_models[degree] = w_poly

table3 = pd.DataFrame(poly_results)
table3_sorted = table3.sort_values(by=["测试MSE", "测试R2", "阶数"], ascending=[True, False, True])
best_degree = int(table3_sorted.iloc[0]["阶数"])
best_poly_model = poly_models[best_degree]

best_poly_row = table3_sorted.iloc[0]
best_poly_train_mse = float(best_poly_row["训练MSE"])
best_poly_test_mse = float(best_poly_row["测试MSE"])
best_poly_train_r2 = float(best_poly_row["训练R2"])
best_poly_test_r2 = float(best_poly_row["测试R2"])

# 5. 高斯基函数回归（同时搜索 m 和 sigma）

gauss_results = []
gauss_models = {}

m_list = [5, 10, 15, 20]
sigma_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

for m in m_list:
    centers = np.linspace(x_train.min(), x_train.max(), m).reshape(-1)

    for sigma in sigma_list:
        X_train_g = gaussian_features(x_train, centers, sigma)
        X_test_g = gaussian_features(x_test, centers, sigma)

        w_gauss = least_squares(X_train_g, y_train)

        y_train_pred = X_train_g @ w_gauss
        y_test_pred = X_test_g @ w_gauss

        train_mse, train_r2 = evaluate(y_train, y_train_pred)
        test_mse, test_r2 = evaluate(y_test, y_test_pred)

        gauss_results.append({
            "基函数个数": m,
            "宽度sigma": sigma,
            "训练MSE": train_mse,
            "测试MSE": test_mse,
            "训练R2": train_r2,
            "测试R2": test_r2
        })

        gauss_models[(m, sigma)] = (w_gauss, centers, sigma)

table4 = pd.DataFrame(gauss_results)
# 排序：先看测试MSE，再看测试R2，再看基函数个数
table4_sorted = table4.sort_values(
    by=["测试MSE", "测试R2", "基函数个数"],
    ascending=[True, False, True]
)

best_m = int(table4_sorted.iloc[0]["基函数个数"])
best_sigma = float(table4_sorted.iloc[0]["宽度sigma"])
best_w_gauss, best_centers, best_sigma = gauss_models[(best_m, best_sigma)]

best_gauss_row = table4_sorted.iloc[0]
best_gauss_train_mse = float(best_gauss_row["训练MSE"])
best_gauss_test_mse = float(best_gauss_row["测试MSE"])
best_gauss_train_r2 = float(best_gauss_row["训练R2"])
best_gauss_test_r2 = float(best_gauss_row["测试R2"])

# 6. 表5：三类回归模型最优结果对比表

linear_train_mse, linear_train_r2 = evaluate(y_train, y_train_ls)
linear_test_mse, linear_test_r2 = evaluate(y_test, y_test_ls)

table5 = pd.DataFrame([
    ["线性回归", "最小二乘法", "-", linear_train_mse, linear_test_mse, linear_train_r2, linear_test_r2],
    ["多项式回归", f"{best_degree}阶", "-", best_poly_train_mse, best_poly_test_mse, best_poly_train_r2, best_poly_test_r2],
    ["高斯基函数回归", f"m={best_m}", f"σ={best_sigma:.2f}", best_gauss_train_mse, best_gauss_test_mse, best_gauss_train_r2, best_gauss_test_r2]
], columns=["模型", "参数1", "参数2", "训练MSE", "测试MSE", "训练R2", "测试R2"])

# 7. 输出结果

print("表1：线性回归模型参数计算结果")
print(table1.round(6))

print("\n表2：线性回归三种方法的性能比较表")
print(table2.round(6))

print("\n表3：不同阶数多项式回归模型结果")
print(table3.round(6))

print("\n表5：三类回归模型最优结果对比表")
print(table5.round(6))

print(f"\n最优多项式阶数：{best_degree}")
print(f"最优高斯基函数参数组合：基函数个数={best_m}, sigma={best_sigma:.6f}")

#绘图准备
x_plot = np.linspace(min(x_train.min(), x_test.min()),
                     max(x_train.max(), x_test.max()), 400).reshape(-1, 1)

#  图1：训练集与测试集散点图

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label="训练集", alpha=0.7)
plt.scatter(x_test, y_test, label="测试集", alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("图1 训练集与测试集散点图")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("图1_训练集与测试集散点图.png", dpi=300)
plt.show()

# 图2：线性回归三种拟合方法结果图

X_plot_lin = add_bias(x_plot)

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label="训练集", alpha=0.5)
plt.scatter(x_test, y_test, label="测试集", alpha=0.5)
plt.plot(x_plot, X_plot_lin @ w_ls, label="最小二乘法", linewidth=2)
plt.plot(x_plot, X_plot_lin @ w_gd, '--', label="梯度下降法", linewidth=2)
plt.plot(x_plot, X_plot_lin @ w_nt, ':', label="牛顿法", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("图2 线性回归三种拟合方法结果图")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("图2_线性回归三种拟合方法结果图.png", dpi=300)
plt.show()


# 图3：最优多项式回归模型拟合结果图

x_plot_poly = 2 * (x_plot - x_min_poly) / (x_max_poly - x_min_poly) - 1
X_plot_best_poly = polynomial_features(x_plot_poly, best_degree)
y_plot_best_poly = X_plot_best_poly @ best_poly_model

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label="训练集", alpha=0.5)
plt.scatter(x_test, y_test, label="测试集", alpha=0.5)
plt.plot(x_plot, y_plot_best_poly, color='red', linewidth=2.5,
         label=f"最优多项式回归（{best_degree}阶）")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"图3 最优多项式回归模型拟合结果图（{best_degree}阶）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("图3_最优多项式回归模型拟合结果图.png", dpi=300)
plt.show()


# 图4：不同基函数个数下训练MSE随σ变化折线图

plt.figure(figsize=(8, 6))
for m in m_list:
    subset = table4[table4["基函数个数"] == m].sort_values("宽度sigma")
    plt.plot(subset["宽度sigma"], subset["训练MSE"],
             marker='o', linewidth=2, label=f"m={m}")

best_train_row = table4.loc[table4["训练MSE"].idxmin()]
plt.scatter(best_train_row["宽度sigma"], best_train_row["训练MSE"],
            color='red', s=100, zorder=5, label=f"最优点(m={int(best_train_row['基函数个数'])}, σ={best_train_row['宽度sigma']:.2f})")

plt.xlabel("宽度 σ")
plt.ylabel("训练MSE")
plt.title("图4 不同基函数个数下训练MSE随σ变化折线图")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("图4_不同基函数个数下训练MSE随σ变化折线图.png", dpi=300)
plt.show()


#  图5：不同基函数个数下训练R2随σ变化折线图

plt.figure(figsize=(8, 6))
for m in m_list:
    subset = table4[table4["基函数个数"] == m].sort_values("宽度sigma")
    plt.plot(subset["宽度sigma"], subset["训练R2"],
             marker='o', linewidth=2, label=f"m={m}")

best_train_r2_row = table4.loc[table4["训练R2"].idxmax()]
plt.scatter(best_train_r2_row["宽度sigma"], best_train_r2_row["训练R2"],
            color='red', s=100, zorder=5, label=f"最优点(m={int(best_train_r2_row['基函数个数'])}, σ={best_train_r2_row['宽度sigma']:.2f})")

plt.xlabel("宽度 σ")
plt.ylabel("训练R2")
plt.title("图5 不同基函数个数下训练R2随σ变化折线图")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("图5_不同基函数个数下训练R2随σ变化折线图.png", dpi=300)
plt.show()


# 图6：不同基函数个数下测试MSE随σ变化折线图

plt.figure(figsize=(8, 6))
for m in m_list:
    subset = table4[table4["基函数个数"] == m].sort_values("宽度sigma")
    plt.plot(subset["宽度sigma"], subset["测试MSE"],
             marker='o', linewidth=2, label=f"m={m}")

plt.scatter(best_sigma, best_gauss_test_mse,
            color='red', s=100, zorder=5, label=f"最优点(m={best_m}, σ={best_sigma:.2f})")

plt.xlabel("宽度 σ")
plt.ylabel("测试MSE")
plt.title("图6 不同基函数个数下测试MSE随σ变化折线图")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("图6_不同基函数个数下测试MSE随σ变化折线图.png", dpi=300)
plt.show()

# 图7：不同基函数个数下测试R2随σ变化折线图

plt.figure(figsize=(8, 6))
for m in m_list:
    subset = table4[table4["基函数个数"] == m].sort_values("宽度sigma")
    plt.plot(subset["宽度sigma"], subset["测试R2"],
             marker='o', linewidth=2, label=f"m={m}")

best_test_r2_row = table4.loc[table4["测试R2"].idxmax()]
plt.scatter(best_test_r2_row["宽度sigma"], best_test_r2_row["测试R2"],
            color='red', s=100, zorder=5, label=f"最优点(m={int(best_test_r2_row['基函数个数'])}, σ={best_test_r2_row['宽度sigma']:.2f})")

plt.xlabel("宽度 σ")
plt.ylabel("测试R2")
plt.title("图7 不同基函数个数下测试R2随σ变化折线图")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("图7_不同基函数个数下测试R2随σ变化折线图.png", dpi=300)
plt.show()


# 图8：最优高斯基函数回归模型拟合结果图

X_plot_best_gauss = gaussian_features(x_plot, best_centers, best_sigma)
y_plot_best_gauss = X_plot_best_gauss @ best_w_gauss

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label="训练集", alpha=0.5)
plt.scatter(x_test, y_test, label="测试集", alpha=0.5)
plt.plot(x_plot, y_plot_best_gauss, color='purple', linewidth=2.5,
         label=f"最优高斯基函数回归（m={best_m}, σ={best_sigma:.3f}）")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"图8 最优高斯基函数回归模型拟合结果图（m={best_m}, σ={best_sigma:.3f}）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("图8_最优高斯基函数回归模型拟合结果图.png", dpi=300)
plt.show()

