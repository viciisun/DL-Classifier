# Deep Learning Classifier: Model Analysis

## Introduction

This document provides a detailed explanation of the neural network classifier implemented for the COMP4329 Deep Learning assignment. The goal is to help you understand the architecture, features, and operations used in the model.

[English]

## Model Overview

This neural network is a feedforward neural network with multiple hidden layers, implemented from scratch using NumPy. It includes several important features that are commonly used in modern neural networks:

1. **Multi-layer Architecture**: The network has an input layer, multiple hidden layers, and an output layer.
2. **Activation Functions**: ReLU (default) or GELU activation functions in the hidden layers.
3. **Regularization**: Dropout and L2 regularization (weight decay) to prevent overfitting.
4. **Batch Normalization**: Normalizes the activations within each mini-batch to improve training stability.
5. **Optimization**: Support for SGD with momentum or Adam optimizer.
6. **Loss Function**: Softmax activation with cross-entropy loss for classification tasks.

## Detailed Components Explanation

### 1. Neural Network Architecture

#### Input Layer

- Takes the input features (e.g., images represented as vectors).
- The default dimension is 128, but this adapts to your input data.

#### Hidden Layers

- The default architecture has two hidden layers with sizes [256, 128].
- Each hidden layer applies a linear transformation (weights and biases), followed by batch normalization (if enabled), activation function, and dropout (during training).

#### Output Layer

- Final layer produces logits that are converted to probabilities using the softmax function.
- For a 10-class classification problem, the output dimension is 10.

### 2. Activation Functions

#### ReLU (Rectified Linear Unit)

- Formula: `f(x) = max(0, x)`
- The default activation function.
- Simple and computationally efficient.
- Helps prevent the vanishing gradient problem.

#### GELU (Gaussian Error Linear Unit)

- Formula: `f(x) = x * Φ(x)` where Φ(x) is the cumulative distribution function of the standard normal distribution.
- A smoother alternative to ReLU.
- Used in modern transformer architectures like BERT.
- Approximated using: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`

### 3. Regularization Techniques

#### Dropout

- Randomly sets a fraction of the inputs to zero during training.
- Default dropout probability is 0.5 (50% of neurons are deactivated).
- Helps prevent overfitting by forcing the network to learn redundant representations.
- During inference (testing), all neurons are active but outputs are scaled to maintain the expected output magnitude.

#### Weight Decay (L2 Regularization)

- Adds a penalty term to the loss function proportional to the square of the weights.
- Default weight decay coefficient is 1e-4.
- Encourages the model to use smaller weights, resulting in a simpler model less prone to overfitting.

### 4. Batch Normalization

- Normalizes the activations of each layer within a mini-batch to have zero mean and unit variance.
- Then applies a scale (gamma) and shift (beta) that are learned during training.
- Formula: `y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta`
- Benefits:
  - Improves gradient flow through the network
  - Allows higher learning rates
  - Reduces dependency on careful initialization
  - Acts as a form of regularization

### 5. Optimization Algorithms

#### SGD with Momentum

- Updates weights using both the current gradient and the previous update.
- Formula: `v = momentum * v - learning_rate * gradient; weights += v`
- Default momentum coefficient is 0.9.
- Helps accelerate training and overcome local minima.

#### Adam Optimizer

- Adaptive optimization algorithm that combines ideas from RMSProp and momentum.
- Maintains first-order (mean) and second-order (uncentered variance) moments of the gradients.
- Applies bias correction to these moments.
- Adaptive learning rate for each parameter.
- Formula:
  - `m = beta1 * m + (1 - beta1) * gradient`
  - `v = beta2 * v + (1 - beta2) * gradient^2`
  - `m_corrected = m / (1 - beta1^t)`
  - `v_corrected = v / (1 - beta2^t)`
  - `weights -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)`

### 6. Loss Function: Softmax and Cross-Entropy

#### Softmax

- Converts raw output scores (logits) to probabilities that sum to 1.
- Formula: `softmax(x)_i = exp(x_i) / sum(exp(x_j))` for all j in the same class.
- Used for multi-class classification problems.

#### Cross-Entropy Loss

- Measures the difference between the predicted probability distribution and the true distribution.
- For one-hot encoded targets, reduces to: `-log(p_y)` where p_y is the predicted probability for the true class.
- Minimizing cross-entropy is equivalent to maximizing the likelihood of the correct class.

### 7. Data Preprocessing

#### Standard Scaling

- Normalizes each feature to have zero mean and unit variance.
- Formula: `x_normalized = (x - mean) / std`
- Helps all features contribute equally to the learning process.

#### Min-Max Scaling

- Scales features to a specific range, typically [0, 1].
- Formula: `x_normalized = (x - min) / (max - min)`
- Preserves relationships between data points.

#### Principal Component Analysis (PCA)

- Reduces dimensionality while retaining most of the variance in the data.
- Projects data onto a lower-dimensional space defined by the principal components.
- Can help reduce noise and computational complexity.

### 8. Training Process

#### Mini-Batch Training

- Divides training data into small batches (default size: 128).
- Updates model parameters after processing each batch.
- Benefits:
  - More efficient than full-batch training
  - Provides more frequent updates than full-batch
  - Introduces some noise which can help escape local minima

#### Forward Pass

1. Input data passes through each layer in sequence.
2. At each hidden layer:
   - Apply linear transformation (weights and biases)
   - Apply batch normalization (if enabled)
   - Apply activation function (ReLU or GELU)
   - Apply dropout (if in training mode)
3. At the output layer:
   - Apply linear transformation
   - Apply softmax to get probability distribution

#### Backward Pass (Backpropagation)

1. Compute the gradient of the loss with respect to the output.
2. Propagate the gradient backwards through the network using the chain rule.
3. Compute gradients with respect to all parameters (weights and biases).
4. Update parameters using the optimizer (SGD with momentum or Adam).

#### Early Stopping

- Stops training when the validation performance ceases to improve.
- Monitors validation accuracy and stops if no improvement for a specified number of epochs (patience).
- Helps prevent overfitting by not training for too long.

### 9. Evaluation Metrics

#### Accuracy

- The proportion of correctly classified instances.
- Formula: `correct predictions / total predictions`
- Simple to understand but can be misleading for imbalanced datasets.

#### Precision

- The proportion of true positive predictions among all positive predictions.
- Formula: `true positives / (true positives + false positives)`
- Measures how many of the predicted positives are actually positive.

#### Recall

- The proportion of true positive predictions among all actual positives.
- Formula: `true positives / (true positives + false negatives)`
- Measures how many of the actual positives were correctly identified.

#### F1 Score

- The harmonic mean of precision and recall.
- Formula: `2 * (precision * recall) / (precision + recall)`
- Provides a balance between precision and recall.

#### Confusion Matrix

- A table showing the counts of true positives, false positives, true negatives, and false negatives.
- Helps visualize the performance of the classifier for each class.

## Runtime Analysis

The implementation monitors and reports various runtime metrics:

1. **Training Time**: The total time taken to train the model.
2. **Epoch Time**: The time taken for each epoch during training.
3. **Inference Time**: The time taken to make predictions on the test data.

These metrics are important for understanding the computational efficiency of the model and can be used to compare different model configurations.

## Ablation Studies

Ablation studies involve systematically removing or replacing components of the model to understand their impact. The main features that can be ablated in this implementation are:

1. **GELU Activation**: Compare performance with ReLU vs. GELU.
2. **Adam Optimizer**: Compare performance with SGD+momentum vs. Adam.
3. **Preprocessing**: Compare performance with different preprocessing methods.
4. **Network Architecture**: Compare performance with different hidden layer configurations.

[中文]

## 模型概述

这个神经网络是一个使用 NumPy 从头实现的具有多个隐藏层的前馈神经网络。它包含了现代神经网络中常用的几个重要特性：

1. **多层架构**：网络有输入层、多个隐藏层和输出层。
2. **激活函数**：在隐藏层中使用 ReLU（默认）或 GELU 激活函数。
3. **正则化**：使用 Dropout 和 L2 正则化（权重衰减）来防止过拟合。
4. **批量归一化**：对每个小批量内的激活进行归一化，以提高训练稳定性。
5. **优化**：支持带动量的 SGD 或 Adam 优化器。
6. **损失函数**：使用 Softmax 激活和交叉熵损失进行分类任务。

## 详细组件说明

### 1. 神经网络架构

#### 输入层

- 接收输入特征（例如表示为向量的图像）。
- 默认维度为 128，但会根据输入数据自适应调整。

#### 隐藏层

- 默认架构有两个隐藏层，大小为[256, 128]。
- 每个隐藏层应用线性变换（权重和偏置），然后是批量归一化（如果启用）、激活函数和 dropout（在训练期间）。

#### 输出层

- 最终层产生通过 softmax 函数转换为概率的 logits。
- 对于 10 类分类问题，输出维度为 10。

### 2. 激活函数

#### ReLU（修正线性单元）

- 公式：`f(x) = max(0, x)`
- 默认激活函数。
- 简单且计算效率高。
- 有助于防止梯度消失问题。

#### GELU（高斯误差线性单元）

- 公式：`f(x) = x * Φ(x)`，其中 Φ(x)是标准正态分布的累积分布函数。
- ReLU 的平滑替代。
- 用于现代 Transformer 架构，如 BERT。
- 近似计算：`0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`

### 3. 正则化技术

#### Dropout

- 在训练期间随机将一部分输入设置为零。
- 默认 dropout 概率为 0.5（50%的神经元被停用）。
- 通过强制网络学习冗余表示来帮助防止过拟合。
- 在推理（测试）期间，所有神经元都处于活动状态，但输出会进行缩放以保持预期的输出幅度。

#### 权重衰减（L2 正则化）

- 向损失函数添加与权重平方成比例的惩罚项。
- 默认权重衰减系数为 1e-4。
- 鼓励模型使用较小的权重，产生更简单的模型，不易过拟合。

### 4. 批量归一化

- 将每层内的激活归一化为具有零均值和单位方差。
- 然后应用在训练期间学习的缩放（gamma）和偏移（beta）。
- 公式：`y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta`
- 好处：
  - 改善网络中的梯度流动
  - 允许更高的学习率
  - 减少对精心初始化的依赖
  - 作为一种形式的正则化

### 5. 优化算法

#### 带动量的 SGD

- 使用当前梯度和先前更新来更新权重。
- 公式：`v = momentum * v - learning_rate * gradient; weights += v`
- 默认动量系数为 0.9。
- 有助于加速训练并克服局部最小值。

#### Adam 优化器

- 结合了 RMSProp 和动量思想的自适应优化算法。
- 维护梯度的一阶（均值）和二阶（非中心方差）矩。
- 对这些矩进行偏差修正。
- 为每个参数提供自适应学习率。
- 公式：
  - `m = beta1 * m + (1 - beta1) * gradient`
  - `v = beta2 * v + (1 - beta2) * gradient^2`
  - `m_corrected = m / (1 - beta1^t)`
  - `v_corrected = v / (1 - beta2^t)`
  - `weights -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)`

### 6. 损失函数：Softmax 和交叉熵

#### Softmax

- 将原始输出分数（logits）转换为总和为 1 的概率。
- 公式：`softmax(x)_i = exp(x_i) / sum(exp(x_j))`，对于同一类中的所有 j。
- 用于多类分类问题。

#### 交叉熵损失

- 测量预测概率分布与真实分布之间的差异。
- 对于独热编码的目标，简化为：`-log(p_y)`，其中 p_y 是真实类的预测概率。
- 最小化交叉熵等同于最大化正确类的似然。

### 7. 数据预处理

#### 标准缩放

- 将每个特征归一化为零均值和单位方差。
- 公式：`x_normalized = (x - mean) / std`
- 帮助所有特征平等地贡献于学习过程。

#### 最小-最大缩放

- 将特征缩放到特定范围，通常为[0, 1]。
- 公式：`x_normalized = (x - min) / (max - min)`
- 保留数据点之间的关系。

#### 主成分分析（PCA）

- 减少维度，同时保留数据中的大部分方差。
- 将数据投影到由主成分定义的低维空间。
- 可以帮助减少噪声和计算复杂性。

### 8. 训练过程

#### 小批量训练

- 将训练数据分成小批量（默认大小：128）。
- 处理每个批次后更新模型参数。
- 好处：
  - 比全批量训练更高效
  - 比全批量提供更频繁的更新
  - 引入一些噪声，可以帮助逃离局部最小值

#### 前向传播

1. 输入数据按顺序通过每一层。
2. 在每个隐藏层：
   - 应用线性变换（权重和偏置）
   - 应用批量归一化（如果启用）
   - 应用激活函数（ReLU 或 GELU）
   - 应用 dropout（如果在训练模式下）
3. 在输出层：
   - 应用线性变换
   - 应用 softmax 获取概率分布

#### 反向传播

1. 计算损失相对于输出的梯度。
2. 使用链式法则将梯度向后传播通过网络。
3. 计算相对于所有参数（权重和偏置）的梯度。
4. 使用优化器（带动量的 SGD 或 Adam）更新参数。

#### 早停

- 当验证性能不再提高时停止训练。
- 监控验证准确率，如果在指定数量的轮次（耐心值）内没有改善就停止。
- 通过不过度训练来帮助防止过拟合。

### 9. 评估指标

#### 准确率

- 正确分类实例的比例。
- 公式：`正确预测 / 总预测`
- 易于理解，但对于不平衡数据集可能具有误导性。

#### 精确率

- 所有正预测中真正例的比例。
- 公式：`真正例 / (真正例 + 假正例)`
- 衡量预测为正的样本中有多少实际为正。

#### 召回率

- 所有实际正例中真正例的比例。
- 公式：`真正例 / (真正例 + 假负例)`
- 衡量实际正例中有多少被正确识别。

#### F1 分数

- 精确率和召回率的调和平均。
- 公式：`2 * (精确率 * 召回率) / (精确率 + 召回率)`
- 提供精确率和召回率之间的平衡。

#### 混淆矩阵

- 显示真正例、假正例、真负例和假负例计数的表格。
- 帮助可视化分类器对每个类的性能。

## 运行时分析

该实现监控并报告各种运行时指标：

1. **训练时间**：训练模型所需的总时间。
2. **轮次时间**：训练期间每个轮次所需的时间。
3. **推理时间**：对测试数据进行预测所需的时间。

这些指标对于理解模型的计算效率很重要，可用于比较不同的模型配置。

## 消融研究

消融研究涉及系统地移除或替换模型的组件以了解它们的影响。此实现中可以进行消融的主要特性是：

1. **GELU 激活**：比较 ReLU 与 GELU 的性能。
2. **Adam 优化器**：比较带动量的 SGD 与 Adam 的性能。
3. **预处理**：比较不同预处理方法的性能。
4. **网络架构**：比较不同隐藏层配置的性能。
