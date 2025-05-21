import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit  # sigmoid 函數

# 橫軸：線性組合 wx + b
x = np.linspace(-10, 10, 200)

# 設定三種不同的截距 b
intercepts = [-5, 0, 5]
colors = ["red", "green", "blue"]
labels = [f"intercept={b}" for b in intercepts]

# 畫圖
plt.figure(figsize=(8, 5))
for b, color, label in zip(intercepts, colors, labels):
    y = expit(x + b)  # 假設 w = 1，線性組合為 x + b
    plt.plot(x, y, label=label, color=color)

# 圖表細節
plt.title("Sigmoid 函數在不同截距下的變化")
plt.xlabel("線性組合 wx + b")
plt.ylabel("機率")
plt.legend()
plt.grid(True)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
