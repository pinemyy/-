import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 导入Keras相关库
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

warnings.filterwarnings('ignore')


# -------------------------- 1. 数据预处理（CNN专用，保持28x28尺寸） --------------------------
def preprocess_cnn_data(raw_data):
    """
    预处理为CNN输入格式：
    - 保持28x28尺寸（原始MNIST尺寸）
    - 反色（统一白底黑字）
    - 归一化到[0,1]
    - 形状调整为(样本数, 28, 28, 1)（单通道灰度图）
    """
    processed_data = []
    raw_images = raw_data.reshape(-1, 28, 28)  # (样本数, 28, 28)
    for img_28x28 in raw_images:
        # 反色（MNIST原始是黑底白字，转为白底黑字）
        img_inverted = 255 - img_28x28
        # 归一化
        img_normalized = img_inverted / 255.0
        processed_data.append(img_normalized)
    # 调整形状为(样本数, 28, 28, 1)
    return np.array(processed_data).reshape(-1, 28, 28, 1)


# -------------------------- 2. CNN模型定义 --------------------------
def build_cnn_model():
    """构建简单CNN模型（适合MNIST数据集）"""
    model = Sequential([
        # 卷积层1：32个3x3卷积核，ReLU激活
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # 池化层1：2x2最大池化
        MaxPooling2D((2, 2)),
        # 卷积层2：64个3x3卷积核，ReLU激活
        Conv2D(64, (3, 3), activation='relu'),
        # 池化层2：2x2最大池化
        MaxPooling2D((2, 2)),
        # 卷积层3：64个3x3卷积核，ReLU激活
        Conv2D(64, (3, 3), activation='relu'),
        # 展平为一维向量
        Flatten(),
        # 全连接层：64个神经元，ReLU激活
        Dense(64, activation='relu'),
        # Dropout防止过拟合
        Dropout(0.5),
        # 输出层：10个神经元（0-9），softmax激活
        Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # 多分类交叉熵损失
        metrics=['accuracy']
    )
    return model


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
    X_train_raw, X_test_raw = X_raw[:60000], X_raw[60000:]
    y_train, y_test = y_raw[:60000], y_raw[60000:]

    # 3. 预处理（CNN专用）
    print("\n3. 预处理（28×28→归一化→调整维度）...")
    X_train = preprocess_cnn_data(X_train_raw)
    X_test = preprocess_cnn_data(X_test_raw)
    # 标签独热编码（适应categorical_crossentropy损失）
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)
    print(f"   训练集形状：{X_train.shape} | 测试集形状：{X_test.shape}")

    # 4. 构建并训练CNN模型
    print("\n4. 构建并训练CNN模型...")
    model = build_cnn_model()
    model.summary()  # 打印模型结构
    # 训练（10轮，批次大小32）
    history = model.fit(
        X_train, y_train_onehot,
        epochs=10,
        batch_size=32,
        validation_split=0.1  # 用10%训练集做验证
    )

    # 5. 评估测试集准确率（预期95%+）
    print("\n5. 评估测试集准确率...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # 取概率最大的类别
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   测试集准确率：{accuracy:.4f}（CNN正常范围：95%-99%）")

    # 6. 保存模型（供app.py加载）
    model_save_path = "digit_cnn_model.h5"
    model.save(model_save_path)
    print(f"\n6. 模型已保存到：{model_save_path}")
    print("=" * 50)
    print("✅ CNN模型训练完成！可在app.py中加载使用")


if __name__ == "__main__":
    main()