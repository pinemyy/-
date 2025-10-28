import numpy as np
from PIL import Image
import pickle
import warnings
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 数据预处理（适合多层感知机的连续值处理）
def batch_preprocess_mnist(raw_data, target_size=(28, 28)):
    processed_data = []
    raw_images = raw_data.reshape(-1, 28, 28)
    for img_28x28 in raw_images:
        img_pil = Image.fromarray(img_28x28.astype(np.uint8))
        img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
        img_inverted = 255 - np.array(img_resized)
        # 使用连续值而不是二值化，并归一化到[0,1]
        img_normalized = img_inverted.astype(np.float32) / 255.0
        processed_data.append(img_normalized.flatten())
    return np.array(processed_data)

# 多层感知机模型类（优化参数适合数字识别）
class MultiLayerPerceptron:
    def __init__(self, hidden_layer_sizes=(256, 128), learning_rate=0.01, max_iter=300):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            solver='adam',  # 使用adam优化器
            activation='relu',  # 使用relu激活函数
            alpha=0.0001,  # L2正则化
            batch_size=200
        )
        # 由于数据已经归一化到[0,1]，不需要额外标准化
        self.scaler = None

    def fit(self, X_train, y_train):
        # 数据已经归一化，直接训练
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def train(self, X_train, y_train):
        """为了保持与原有接口兼容，添加train方法"""
        return self.fit(X_train, y_train)

# 训练流程
def main():
    print("=" * 50)
    # 1. 加载MNIST数据集
    print("1. 加载MNIST手写数字数据集...")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X_raw = X_raw.astype(np.uint8)
    y_raw = y_raw.astype(np.uint8)

    # 2. 划分训练集和测试集
    print("\n2. 划分训练集/测试集...")
    X_train_raw, y_train = X_raw[:60000], y_raw[:60000]
    X_test_raw, y_test = X_raw[60000:], y_raw[60000:]

    # 3. 预处理（适合多层感知机的连续值处理）
    print("\n3. 批量预处理（28×28→归一化到[0,1]）...")
    X_train = batch_preprocess_mnist(X_train_raw)
    X_test = batch_preprocess_mnist(X_test_raw)

    # 4. 训练多层感知机模型
    print("\n4. 训练多层感知机模型...")
    model = MultiLayerPerceptron(
        hidden_layer_sizes=(256, 128),  # 两层隐藏层：256个神经元，128个神经元
        learning_rate=0.01,
        max_iter=300
    )
    print("   开始训练...")
    model.train(X_train, y_train)
    print("   训练完成！")

    # 5. 评估准确率
    print("\n5. 评估测试集准确率...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   测试集准确率：{accuracy:.4f}（多层感知机正常范围：85%-95%）")

    # 6. 保存模型
    model_save_path = "digit_perceptron_model.pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n6. 模型已保存到：{model_save_path}")
    print("=" * 50)
    print("✅ 多层感知机训练完成！可在app.py中加载使用")

if __name__ == "__main__":
    main()