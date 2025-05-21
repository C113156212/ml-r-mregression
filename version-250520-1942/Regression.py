import numpy as np
import matplotlib.pyplot as plt
import os
import json

# 1. 模擬資料
X = np.array([[30, 1], [45, 1], [50, 2], [55, 2], [65, 2],
              [70, 3], [80, 3], [85, 3], [90, 3], [95, 3]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# 2. 設定超參數
lr = 0.003
epochs = 10000
loss_list = []

# 儲存參數的檔案
model_file = "Machine_Learning_Models/a_Supervised_Learning/Regression/mRegression/version/version-250520-1942/logistic_params.json"

# 3. 嘗試讀取已儲存的模型參數
if os.path.exists(model_file):
    with open(model_file, "r") as f:
        saved = json.load(f)
        w = np.array(saved["weights"])
        b = saved["bias"]
        print("✅ 已載入模型參數：")
        print("w =", w)
        print("b =", b)
else:
    # 4. 初始化
    print("❌ 模型參數不存在，將使用初始化")
    w = np.zeros(2)
    b = 0
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 5. 開始訓練
for epoch in range(epochs):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    error = y_hat - y

    # 梯度
    dw = np.dot(X.T, error) / len(y)
    db = np.sum(error) / len(y)

    # 更新參數
    w -= lr * dw
    b -= lr * db

    # 記錄損失
    loss = -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
    loss_list.append(loss)

    # 顯示進度條
    if epoch % 100 == 0:
        print(f"第 {epoch} 次訓練 - Loss: {loss:.4f}")

print("訓練完成！")
print("最終權重：", w)
print("最終截距：", b)

# 6. 儲存參數
with open(model_file, "w") as f:
    json.dump({"weights": w.tolist(), "bias": b}, f)
    print("✅ 模型參數已儲存至", model_file)

# 7. 畫圖
loss_list = np.array(loss_list).flatten()  # 強制轉成一維

plt.plot(loss_list)
plt.title("Loss 曲線")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

