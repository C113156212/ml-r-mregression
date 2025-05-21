# 邏輯回歸分類模型專案
# 目標：根據筆試分數與面試表現預測是否錄取（分類問題）

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 設定隨機種子以便重現
np.random.seed(42)

# 1. 生成 100 筆訓練資料
# 分數區間 40-99，面試表現 1-5
scores = np.random.randint(40, 100, 100)
performance = np.random.randint(1, 6, 100)

# 自訂錄取邏輯：分數與表現的加權總分超過門檻就錄取
admitted = (scores * 0.6 + performance * 10 > 100).astype(int)

# 整合成 DataFrame
X = pd.DataFrame({
    "score": scores,
    "performance": performance
})
y = admitted

# 2. 建立邏輯回歸模型
model = LogisticRegression()

# 3. 使用 5 折交叉驗證評估模型
cv_scores = cross_val_score(model, X, y, cv=5)

# 顯示每一折準確率與平均分數
print("每一折準確率：", cv_scores)
print("平均準確率：", round(cv_scores.mean(), 2))

# 4. 使用整體資料訓練模型（準備預測新資料）
model.fit(X, y)

# 5. 新的一筆資料進行預測
new_applicant = pd.DataFrame({
    "score": [75],
    "performance": [3]
})

# 預測分類與機率
pred = model.predict(new_applicant)[0]
proba = model.predict_proba(new_applicant)[0]

# 顯示預測結果
print("\n新申請者資料：")
print(new_applicant)
print("\n預測是否錄取：", pred)
print("預測機率：[不錄取, 錄取]：", np.round(proba, 3))

# 6. 顯示模型的權重（用於解釋模型邏輯）
print("\n模型權重 (score, performance)：", model.coef_)
print("模型截距：", model.intercept_)
