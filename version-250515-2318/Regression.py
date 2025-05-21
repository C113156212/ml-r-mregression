import numpy as np

# 模擬資料（分數+表現，標籤為是否錄取）
X = np.array([[30, 1], [45, 1], [50, 2], [55, 2], [65, 2],
              [70, 3], [80, 3], [85, 3], [90, 3], [95, 3]]) #10組資料
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# 初始化參數
w = np.zeros(2)    # 兩個特徵
b = 0
lr = 0.0031          # 學習率*(初始設置為0.01，但太大了，使得損失函數震盪，且不收斂) --> 所以改成 0.0031
epochs = 5000        # 訓練次數(初始值為1000),但無法收斂 --> 改成5000

# Sigmoid 函數
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 訓練
for epoch in range(epochs):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    error = y_hat - y
    #error= y^−y=[0.956−1,0.9993−0]=[−0.044,0.9993]

    # 梯度計算
    dw = np.dot(X.T, error) / len(y) #X.T 是 X的轉置矩陣
    db = np.sum(error) / len(y)
    
    # 參數更新
    w -= lr * dw
    b -= lr * db

    # 每100次印一次損失
    if epoch % 100 == 0:
        loss = -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        print(f"第 {epoch} 次訓練 - Loss: {loss:.4f}")

print("訓練完成！")
print("最終權重：", w)
print("最終截距：", b)
