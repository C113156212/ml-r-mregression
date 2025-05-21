import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 建立模擬資料
data = {
    "score": [30, 45, 50, 55, 65, 70, 80, 85, 90, 95],
    "performance": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    "admitted": [1,1,1,1,0,0,0,0,0,0]#[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 2. 定義特徵 X 和標籤 y
X = df[["score", "performance"]]  # 兩個輸入變數
y = df["admitted"]               # 目標變數（0 or 1）

# 3. 將資料切分為訓練集和測試集（80% 訓練，20% 測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立邏輯回歸模型
model = LogisticRegression()

# 5. 用訓練資料訓練模型
model.fit(X_train, y_train)

# 6. 用測試資料進行預測
y_pred = model.predict(X_test)

# 7. 輸出準確率
accuracy = accuracy_score(y_test, y_pred)
print("模型準確率：", accuracy)

# 8. 顯示測試資料的預測結果
results = pd.DataFrame({
    "score": X_test["score"],
    "performance": X_test["performance"],
    "實際錄取": y_test,
    "預測錄取": y_pred
})
print(results)
