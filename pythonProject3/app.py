from datetime import datetime  # 正确导入方式
import io
import os
import pickle
import base64
import json  # 确保导入json模块
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from qa_system import qa_system

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ---------------------- 多层感知机模型类定义 ----------------------
class MultiLayerPerceptron:
    def __init__(self, hidden_layer_sizes=(256, 128), learning_rate=0.01, max_iter=300):
        from sklearn.neural_network import MLPClassifier
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

# ---------------------- 朴素贝叶斯模型类定义 ----------------------
class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_prior = None
        self.feature_prob = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape
        self.class_prior = np.zeros(n_classes)
        for idx, cls in enumerate(self.classes):
            cls_count = np.sum(y == cls)
            self.class_prior[idx] = (cls_count + self.alpha) / (n_samples + self.alpha * n_classes)
        self.feature_prob = np.zeros((n_classes, n_features))
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            cls_sample_count = X_cls.shape[0]
            feature_1_count = np.sum(X_cls, axis=0)
            self.feature_prob[idx] = (feature_1_count + self.alpha) / (cls_sample_count + 2 * self.alpha)

    def predict(self, X):
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

    def predict_proba(self, X):
        """返回每个类别的概率"""
        probabilities = []
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
            
            # 转换为概率
            posteriors = np.array(posteriors)
            exp_posteriors = np.exp(posteriors - np.max(posteriors))  # 防止溢出
            proba = exp_posteriors / np.sum(exp_posteriors)
            probabilities.append(proba)
        return np.array(probabilities)


# 加载模型
perceptron_model = None
nb_model = None
cnn_model = None  # CNN模型

# 加载感知机模型
try:
    with open('digit_perceptron_model.pkl', 'rb') as f:
        perceptron_model = pickle.load(f)
    print("✅ 感知机模型加载成功")
except FileNotFoundError:
    print("❌ 未找到感知机模型文件，请先训练模型")
except Exception as e:
    print(f"❌ 感知机模型加载失败：{str(e)}")

# 加载朴素贝叶斯模型
try:
    with open('digit_naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    print("✅ 朴素贝叶斯模型加载成功")
except FileNotFoundError:
    print("❌ 未找到朴素贝叶斯模型文件，请先训练模型")
except Exception as e:
    print(f"❌ 朴素贝叶斯模型加载失败：{str(e)}")

# 加载CNN模型
try:
    cnn_model = load_model('digit_cnn_model.h5')  # Keras模型格式为.h5
    print("✅ CNN模型加载成功")
except FileNotFoundError:
    print("❌ 未找到CNN模型文件，请先训练模型")
except Exception as e:
    print(f"❌ CNN模型加载失败：{str(e)}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def base64_to_image(base64_string):
    """将base64字符串转换为PIL Image对象"""
    try:
        # 移除dataURL前缀（如果存在）
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Base64转换失败: {e}")
        return None


# CNN专用预处理（保持28x28尺寸，适应CNN输入）
def preprocess_image_cnn(img):
    """预处理为28x28灰度图→反色→归一化"""
    # 转为灰度图
    img_gray = img.convert('L')
    # 调整为28x28（CNN输入通常为28x28）
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    # 反色（统一白底黑字）
    img_inverted = 255 - np.array(img_resized)
    # 归一化到[0,1]（CNN常用输入格式）
    img_normalized = img_inverted / 255.0
    # 调整形状为(1, 28, 28, 1)（匹配CNN输入：样本数×高×宽×通道数）
    feature = img_normalized.reshape(1, 28, 28, 1)

    # 预处理图转为base64（前端展示）
    buffer = BytesIO()
    Image.fromarray(img_inverted.astype(np.uint8)).save(buffer, format='PNG')
    preprocess_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return feature, preprocess_img_b64


# 多层感知机专用预处理（28x28归一化数据）
def preprocess_image_mlp(img):
    """预处理为28x28归一化特征（适合多层感知机）"""
    img_gray = img.convert('L')
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    img_inverted = 255 - np.array(img_resized)
    # 归一化到[0,1]（与训练时保持一致）
    img_normalized = img_inverted.astype(np.float32) / 255.0
    feature = img_normalized.flatten()
    
    buffer = BytesIO()
    Image.fromarray(img_inverted.astype(np.uint8)).save(buffer, format='PNG')
    preprocess_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return feature, preprocess_img_b64

# 原有预处理函数（用于朴素贝叶斯）
def preprocess_image(img):
    """预处理为8x8二值特征（用于朴素贝叶斯）"""
    img_gray = img.convert('L')
    img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)
    img_inverted = 255 - np.array(img_resized)
    img_binarized = np.where(img_inverted > 127, 1, 0).flatten()
    buffer = BytesIO()
    Image.fromarray(img_inverted.astype(np.uint8)).save(buffer, format='PNG')
    preprocess_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_binarized, preprocess_img_b64


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/rec')
def rec():
    return render_template('rec.html')


@app.route('/cnn_1')
def cnn_1():
    return render_template('cnn_1.html')


@app.route('/naive_1')
def naive_1():
    return render_template('naive_1.html')


@app.route('/perp')
def perp():
    return render_template('perp.html')


@app.route('/recognition')
def recognition():
    return render_template('recognition.html', preprocess_img=None, result=None)


@app.route('/recognition/naive_bayes')
def naive_bayes_recognition():
    return render_template('naive_bayes_recognition.html', preprocess_img=None, result=None)


@app.route('/recognition/cnn')
def cnn_recognition():
    return render_template('cnn_recognition.html', preprocess_img=None, result=None)


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/qa')
def qa():
    """智能问答页面"""
    return render_template('qa.html')


@app.route('/api/qa/chat', methods=['POST'])
def qa_chat():
    """智能问答聊天接口"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': '未提供消息内容'
            })
        
        user_message = data['message']
        user_id = data.get('user_id', 'anonymous')
        
        # 处理用户消息
        response = qa_system.process_message(user_message, user_id)
        
        return jsonify({
            'success': True,
            'response': response['response'],
            'intent': response['intent'],
            'dialogue_state': response['dialogue_state'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"智能问答接口错误: {e}")
        return jsonify({
            'success': False,
            'error': f'处理失败: {str(e)}'
        })


@app.route('/api/qa/reset', methods=['POST'])
def qa_reset():
    """重置问答会话"""
    try:
        qa_system.reset_session()
        return jsonify({
            'success': True,
            'message': '会话已重置'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'重置失败: {str(e)}'
        })

# 手写数字ai接口
@app.route('/api/online-recognize', methods=['POST'])
def online_recognize():
    """在线手写识别接口 - 适配前端画布图像"""
    try:
        # 获取前端发送的JSON数据
        data = request.get_json()
        print("=== 收到识别请求 ===")
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': '未提供图像数据'
            })

        # 提取base64图像数据
        image_data = data['image']
        print(f"图像数据长度: {len(image_data)}")
        
        # 转换base64为PIL Image对象
        image = base64_to_image(image_data)
        if image is None:
            return jsonify({
                'success': False,
                'error': '图像数据格式错误'
            })

        print(f"图像尺寸: {image.size}")

        # 初始化预测结果
        predictions = {
            'perceptron': -1,
            'naive_bayes': -1,
            'cnn': -1,
            'cnn_confidence': 0
        }

        # 感知机识别
        if perceptron_model:
            try:
                print("开始感知机识别...")
                feature, _ = preprocess_image_mlp(image)  # 使用MLP专用预处理
                perceptron_result = perceptron_model.predict([feature])[0]
                predictions['perceptron'] = int(perceptron_result)
                print(f"感知机识别结果: {perceptron_result}")
            except Exception as e:
                print(f"感知机识别失败: {e}")
                predictions['perceptron'] = -1

        # 朴素贝叶斯识别
        if nb_model:
            try:
                print("开始朴素贝叶斯识别...")
                feature, _ = preprocess_image(image)
                nb_result = nb_model.predict([feature])[0]
                predictions['naive_bayes'] = int(nb_result)
                print(f"朴素贝叶斯识别结果: {nb_result}")
            except Exception as e:
                print(f"朴素贝叶斯识别失败: {e}")
                predictions['naive_bayes'] = -1

        # CNN识别
        if cnn_model:
            try:
                print("开始CNN识别...")
                feature, _ = preprocess_image_cnn(image)
                cnn_prediction = cnn_model.predict(feature, verbose=0)
                cnn_result = int(np.argmax(cnn_prediction, axis=1)[0])
                cnn_confidence = float(np.max(cnn_prediction))
                
                predictions['cnn'] = cnn_result
                predictions['cnn_confidence'] = round(cnn_confidence * 100, 2)
                # 返回完整概率分布（百分比，保留两位小数）
                try:
                    predictions['cnn_probs'] = [round(float(p) * 100, 2) for p in cnn_prediction[0]]
                except Exception as e:
                    print(f"格式化CNN概率分布失败: {e}")
                    predictions['cnn_probs'] = None
                print(f"CNN识别结果: {cnn_result}, 置信度: {cnn_confidence:.2%}")
            except Exception as e:
                print(f"CNN识别失败: {e}")
                predictions['cnn'] = -1
                predictions['cnn_confidence'] = 0
                predictions['cnn_probs'] = None

        # 计算最终结果（投票机制）
        valid_predictions = []
        if predictions['perceptron'] != -1:
            valid_predictions.append(predictions['perceptron'])
        if predictions['naive_bayes'] != -1:
            valid_predictions.append(predictions['naive_bayes'])
        if predictions['cnn'] != -1:
            valid_predictions.append(predictions['cnn'])

        if valid_predictions:
            # 使用投票机制决定最终结果
            from collections import Counter
            counter = Counter(valid_predictions)
            final_result = counter.most_common(1)[0][0]
            print(f"投票结果: {dict(counter)}, 最终结果: {final_result}")
        else:
            final_result = -1
            print("所有模型识别失败")

        # 构建响应数据
        response_data = {
            'success': True,
            'predictions': predictions,
            'final_result': final_result,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print("=== 识别完成 ===")
        return jsonify(response_data)

    except Exception as e:
        print(f"在线识别接口错误: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'识别失败: {str(e)}'
        })


def base64_to_image(base64_string):
    """将base64字符串转换为PIL Image对象"""
    try:
        # 移除dataURL前缀（如果存在）
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        print(f"成功转换图像: {image.size}, 模式: {image.mode}")
        return image
        
    except Exception as e:
        print(f"Base64转换失败: {e}")
        return None


# # 如果需要，添加调试接口
# @app.route('/api/debug', methods=['POST'])
# def debug_info():
#     """调试接口，检查数据传输"""
#     try:
#         data = request.get_json()
#         print("=== 调试信息 ===")
#         print(f"收到数据类型: {type(data)}")
        
#         if data and 'image' in data:
#             image_data = data['image']
#             print(f"图像数据长度: {len(image_data)}")
#             print(f"图像数据前缀: {image_data[:100]}...")
            
#             # 尝试转换图像
#             image = base64_to_image(image_data)
#             if image:
#                 return jsonify({
#                     'success': True,
#                     'message': f'图像转换成功: {image.size}',
#                     'image_size': f'{image.size}'
#                 })
#             else:
#                 return jsonify({
#                     'success': False,
#                     'message': '图像转换失败'
#                 })
        
#         return jsonify({
#             'success': False,
#             'message': '未收到有效数据'
#         })
        
#     except Exception as e:
#         print(f"调试接口错误: {e}")
#         return jsonify({
#             'success': False,
#             'message': f'调试错误: {str(e)}'
#         })


# # 健康检查接口
# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """健康检查接口"""
#     models_status = {
#         'perceptron': perceptron_model is not None,
#         'naive_bayes': nb_model is not None,
#         'cnn': cnn_model is not None
#     }
    
#     return jsonify({
#         'status': 'healthy',
#         'models': models_status,
#         'timestamp': datetime.now().isoformat()
#     })

@app.route('/process-image', methods=['POST'])
def process_image():
    """感知机处理"""
    if perceptron_model is None:
        return render_template('recognition.html', preprocess_img=None, result="❌ 感知机模型未加载", probabilities=None)
    if 'image' not in request.files:
        return render_template('recognition.html', preprocess_img=None, result="❌ 未选择图片", probabilities=None)
    file = request.files['image']
    if file.filename == '':
        return render_template('recognition.html', preprocess_img=None, result="❌ 文件名不能为空", probabilities=None)
    if file and allowed_file(file.filename):
        try:
            img = Image.open(file.stream)
            feature, preprocess_img = preprocess_image_mlp(img)  # 使用MLP专用预处理
            result = perceptron_model.predict([feature])[0]
            
            # 获取概率信息
            probabilities = None
            if hasattr(perceptron_model.model, 'predict_proba'):
                proba = perceptron_model.model.predict_proba([feature])[0]
                probabilities = {str(i): float(proba[i]) for i in range(10)}
            
            return render_template('recognition.html', preprocess_img=preprocess_img, result=result, probabilities=probabilities)
        except Exception as e:
            return render_template('recognition.html', preprocess_img=None, result=f"❌ 处理失败：{str(e)}", probabilities=None)
    return render_template('recognition.html', preprocess_img=None, result="❌ 不支持的文件格式", probabilities=None)


@app.route('/process-image-naive-bayes', methods=['POST'])
def process_image_naive_bayes():
    """朴素贝叶斯处理"""
    if nb_model is None:
        return render_template('naive_bayes_recognition.html', preprocess_img=None, result="❌ 朴素贝叶斯模型未加载", probabilities=None)
    if 'image' not in request.files:
        return render_template('naive_bayes_recognition.html', preprocess_img=None, result="❌ 未选择图片", probabilities=None)
    file = request.files['image']
    if file.filename == '':
        return render_template('naive_bayes_recognition.html', preprocess_img=None, result="❌ 文件名不能为空", probabilities=None)
    if file and allowed_file(file.filename):
        try:
            img = Image.open(file.stream)
            feature, preprocess_img = preprocess_image(img)
            result = nb_model.predict([feature])[0]
            
            # 获取概率信息
            probabilities = None
            if hasattr(nb_model, 'predict_proba'):
                proba = nb_model.predict_proba([feature])[0]
                probabilities = {str(int(nb_model.classes[i])): float(proba[i]) for i in range(len(nb_model.classes))}
            
            return render_template('naive_bayes_recognition.html', preprocess_img=preprocess_img, result=result, probabilities=probabilities)
        except Exception as e:
            return render_template('naive_bayes_recognition.html', preprocess_img=None, result=f"❌ 处理失败：{str(e)}", probabilities=None)
    return render_template('naive_bayes_recognition.html', preprocess_img=None, result="❌ 不支持的文件格式", probabilities=None)


# CNN处理路由
@app.route('/process-image-cnn', methods=['POST'])
def process_image_cnn():
    """CNN模型识别处理"""
    if cnn_model is None:
        return render_template('cnn_recognition.html', preprocess_img=None, result="❌ CNN模型未加载", probabilities=None)
    if 'image' not in request.files:
        return render_template('cnn_recognition.html', preprocess_img=None, result="❌ 未选择图片", probabilities=None)
    file = request.files['image']
    if file.filename == '':
        return render_template('cnn_recognition.html', preprocess_img=None, result="❌ 文件名不能为空", probabilities=None)
    if file and allowed_file(file.filename):
        try:
            img = Image.open(file.stream)
            # 使用CNN专用预处理（28x28尺寸）
            feature, preprocess_img = preprocess_image_cnn(img)
            # CNN预测（返回概率最大的类别）
            prediction = cnn_model.predict(feature, verbose=0)
            result = np.argmax(prediction, axis=1)[0]
            
            # 获取概率信息
            probabilities = {str(i): float(prediction[0][i]) for i in range(10)}
            
            return render_template('cnn_recognition.html', preprocess_img=preprocess_img, result=result, probabilities=probabilities)
        except Exception as e:
            return render_template('cnn_recognition.html', preprocess_img=None, result=f"❌ 处理失败：{str(e)}", probabilities=None)

    return render_template('cnn_recognition.html', preprocess_img=None, result="❌ 不支持的文件格式", probabilities=None)


if __name__ == '__main__':
    app.run(debug=True)