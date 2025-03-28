import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 加载数据
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列作为特征
y = data.iloc[:, -1]   # 所有行，最后一列作为目标值

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 保存模型
joblib.dump(model, 'linear_model.pkl')

# 保存参数为文本文件
with open('linear_model.txt', 'w') as f:
    f.write(f'系数: {model.coef_}\n截距: {model.intercept_}\n')
