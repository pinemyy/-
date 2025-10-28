# 数字识别系统 - 概率显示功能

## 新增功能

### 1. 概率分布显示
为三个识别模型（感知机、朴素贝叶斯、CNN）添加了概率分布柱状图显示功能：

- **水平柱状图**：显示每个数字（0-9）的识别概率
- **颜色区分**：
  - 感知机：蓝色柱状图
  - 朴素贝叶斯：绿色柱状图  
  - CNN：紫色柱状图
- **动画效果**：柱状图加载时有平滑的过渡动画
- **百分比显示**：每个柱状图上方显示具体的概率百分比

### 2. 后端改进

#### 感知机模型
- 使用 `predict_proba()` 方法获取每个数字的概率
- 返回格式：`{'0': 0.05, '1': 0.02, ..., '9': 0.15}`

#### 朴素贝叶斯模型
- 添加了 `predict_proba()` 方法
- 手动计算后验概率并转换为标准概率分布
- 防止数值溢出，使用对数空间计算

#### CNN模型
- 直接使用模型输出的概率分布
- 返回每个数字的softmax概率

### 3. 前端界面

#### 概率分布图组件
```html
<!-- 概率分布图 -->
{% if probabilities %}
<div class="mt-8 bg-white rounded-lg shadow p-6">
  <h3 class="text-xl font-semibold mb-4 text-gray-800">识别概率分布</h3>
  <div class="space-y-3">
    {% for digit in range(10) %}
    <div class="flex items-center">
      <div class="w-8 text-center font-semibold text-gray-700">{{ digit }}</div>
      <div class="flex-1 mx-3">
        <div class="bg-gray-200 rounded-full h-6 relative">
          <div 
            class="bg-blue-500 h-6 rounded-full transition-all duration-500 ease-out"
            style="width: {{ (probabilities[digit|string] * 100)|round(1) }}%"
          ></div>
          <div class="absolute inset-0 flex items-center justify-center text-xs font-medium text-gray-700">
            {{ (probabilities[digit|string] * 100)|round(1) }}%
          </div>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  <div class="mt-4 text-sm text-gray-600">
    <p>最高概率：数字 <span class="font-bold text-blue-600">{{ result }}</span> ({{ (probabilities[result|string] * 100)|round(1) }}%)</p>
  </div>
</div>
{% endif %}
```

### 4. 技术特点

#### 响应式设计
- 柱状图在不同屏幕尺寸下都能正常显示
- 移动端友好的布局

#### 视觉效果
- 平滑的CSS过渡动画
- 颜色编码区分不同模型
- 清晰的百分比显示

#### 数据处理
- 概率值四舍五入到小数点后1位
- 自动处理缺失的概率数据
- 防止除零错误

### 5. 使用方法

1. 上传图片到任意识别页面
2. 点击"上传并识别"按钮
3. 查看识别结果和概率分布图
4. 概率分布图会显示在结果下方

### 6. 文件修改列表

#### 后端文件
- `app.py`: 修改了三个处理函数，添加概率计算和返回
- `qa_system.py`: 智能问答系统（新增）

#### 前端文件
- `templates/recognition.html`: 感知机识别页面
- `templates/naive_bayes_recognition.html`: 朴素贝叶斯识别页面
- `templates/cnn_recognition.html`: CNN识别页面
- `templates/qa.html`: 智能问答页面（新增）
- `templates/index.html`: 主页面（添加智能问答入口）

#### 测试文件
- `test_probability.py`: 概率功能测试脚本

### 7. 注意事项

1. 确保所有模型都已正确训练和加载
2. 概率计算可能需要一些时间，特别是朴素贝叶斯模型
3. 如果模型不支持概率预测，概率分布图将不会显示
4. 概率值总和应该接近1.0（100%）

### 8. 未来改进

1. 添加概率分布的交互式图表（使用Chart.js等）
2. 支持概率阈值的自定义设置
3. 添加概率分布的历史记录功能
4. 支持批量图片的概率分析
