import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# --- 1. 生成模拟数据 ---
# 创建一个包含1000个样本的二分类问题数据集。
# n_features=10: 每个样本有10个特征。
# n_informative=5: 其中5个特征是有效的（包含信息）。
# random_state: 确保每次运行代码生成的数据都一样，便于复现。
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5, 
    n_redundant=0, 
    n_classes=2, 
    random_state=42
)

print(f"生成的数据维度: {X.shape}")
print(f"生成的标签维度: {y.shape}\n")

# --- 2. 划分训练集和测试集 ---
# 将数据按80:20的比例划分为训练集和测试集。
# test_size=0.2: 测试集占20%。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}\n")

# --- 3. 训练 XGBoost 模型 ---
# 初始化一个 XGBoost 分类器。
# use_label_encoder=False: 推荐设置，避免未来版本中的警告。
# eval_metric='logloss': 指定评估指标为对数损失。
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

print("--- 开始训练 XGBoost 模型 ---")
# 使用训练数据来训练模型
model.fit(X_train, y_train)
print("--- 模型训练完成 ---\n")

# --- 4. 在测试集上进行预测 ---
print("--- 在测试集上进行预测 ---")
y_pred = model.predict(X_test)

# --- 5. 评估模型性能 ---
# 计算并打印模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率 (Accuracy): {accuracy:.4f}\n")

# 打印更详细的分类报告，包括精确率、召回率和F1分数
print("--- 分类报告 (Classification Report) ---")
print(classification_report(y_test, y_pred))

# 打印模型最重要的几个特征
# 这在金融等领域很有用，可以知道哪些因子对预测最重要
print("--- 特征重要性 (Feature Importance) ---")
feature_importances = model.feature_importances_
for i, importance in enumerate(feature_importances):
    print(f"特征 {i}: {importance:.4f}")
