import numpy as np
from PIL import Image
import pickle
import warnings
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# -------------------------- 1. 数据预处理 --------------------------
def batch_preprocess_mnist(raw_data, target_size=(8, 8)):
    """批量预处理MNIST数据：28x28→8x8→反色→二值化"""
    processed_data = []
    raw_images = raw_data.reshape(-1, 28, 28)  # (样本数, 28, 28)
    for img_28x28 in raw_images:
        img_pil = Image.fromarray(img_28x28.astype(np.uint8))
        img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
        img_inverted = 255 - np.array(img_resized)  # 反色（统一白底黑字）
        img_binarized = np.where(img_inverted > 127, 1, 0).flatten()  # 二值化+展平（64维）
        processed_data.append(img_binarized)
    return np.array(processed_data)

# -------------------------- 2. 伯努利朴素贝叶斯模型 --------------------------
class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 拉普拉斯平滑系数
        self.classes = None  # 类别（0-9）
        self.class_prior = None  # 类先验概率 P(数字)
        self.feature_prob = None  # 特征条件概率 P(特征=1 | 数字)

    def fit(self, X, y):
        """训练：统计先验概率和特征条件概率"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape

        # 计算类先验概率
        self.class_prior = np.zeros(n_classes)
        for idx, cls in enumerate(self.classes):
            cls_count = np.sum(y == cls)
            self.class_prior[idx] = (cls_count + self.alpha) / (n_samples + self.alpha * n_classes)

        # 计算特征条件概率
        self.feature_prob = np.zeros((n_classes, n_features))
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            cls_sample_count = X_cls.shape[0]
            feature_1_count = np.sum(X_cls, axis=0)
            self.feature_prob[idx] = (feature_1_count + self.alpha) / (cls_sample_count + 2 * self.alpha)

        print(f"✅ 朴素贝叶斯训练完成（{n_classes}个类别，{n_features}个二值特征）")

    def predict(self, X):
        """预测：计算后验概率最大值"""
        predictions = []
        for x in X:
            posteriors = []
            for idx, cls in enumerate(self.classes):
                log_prior = np.log(self.class_prior[idx])
                log_likelihood = 0
                for i, feat_val in enumerate(x):
                    prob = self.feature_prob[idx, i]
                    if feat_val == 1:
                        log_likelihood += np.log(prob)
                    else:
                        log_likelihood += np.log(1 - prob)
                log_posterior = log_prior + log_likelihood
                posteriors.append(log_posterior)
            pred_cls = self.classes[np.argmax(posteriors)]
            predictions.append(pred_cls)
        return np.array(predictions)

# -------------------------- 3. 训练流程 --------------------------
def main():
    print("=" * 50)
    # 1. 加载MNIST数据集
    print("1. 加载MNIST手写数字数据集...")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X_raw = X_raw.astype(np.uint8)
    y_raw = y_raw.astype(np.uint8)
    print(f"   数据集规模：{X_raw.shape[0]}个样本，{X_raw.shape[1]}维原始特征（28×28）")

    # 2. 划分训练集（60000）和测试集（10000）
    print("\n2. 划分训练集/测试集...")
    X_train_raw, y_train = X_raw[:60000], y_raw[:60000]
    X_test_raw, y_test = X_raw[60000:], y_raw[60000:]

    # 3. 预处理（与app.py完全一致）
    print("\n3. 批量预处理（28×28→8×8→二值化）...")
    X_train = batch_preprocess_mnist(X_train_raw)
    X_test = batch_preprocess_mnist(X_test_raw)
    print(f"   预处理后：训练集{X_train.shape} | 测试集{X_test.shape}（64=8×8）")

    # 4. 训练模型
    print("\n4. 训练伯努利朴素贝叶斯模型...")
    model = BernoulliNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)

    # 5. 评估准确率
    print("\n5. 评估测试集准确率...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   测试集准确率：{accuracy:.4f}（正常范围：85%-90%）")

    # 6. 保存模型
    model_save_path = "digit_naive_bayes_model.pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n6. 模型已保存到：{model_save_path}")
    print("=" * 50)
    print("✅ 朴素贝叶斯训练完成！可在app.py中加载使用")

if __name__ == "__main__":
    main()