import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=128, hidden_sizes=[256, 128], output_size=10, 
                 use_bn=True, dropout_prob=0.5):
        """初始化神经网络"""
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        self.use_bn = use_bn
        self.dropout_prob = dropout_prob

        # Xavier 初始化权重和偏置
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.layers.append({'W': w, 'b': b})
        
        # 批量归一化参数
        if use_bn:
            for i in range(len(hidden_sizes)):
                self.layers[i]['gamma'] = np.ones((1, sizes[i+1]))
                self.layers[i]['beta'] = np.zeros((1, sizes[i+1]))
                self.layers[i]['running_mean'] = np.zeros((1, sizes[i+1]))
                self.layers[i]['running_var'] = np.ones((1, sizes[i+1]))

        # 动量 SGD 的速度
        self.velocities = [{'W': np.zeros_like(l['W']), 'b': np.zeros_like(l['b'])} 
                          for l in self.layers]
        if use_bn:
            for i in range(len(hidden_sizes)):
                self.velocities[i]['gamma'] = np.zeros_like(self.layers[i]['gamma'])
                self.velocities[i]['beta'] = np.zeros_like(self.layers[i]['beta'])

    def relu(self, x):
        """ReLU 激活函数"""
        return np.maximum(0, x)

    def relu_grad(self, x):
        """ReLU 的梯度"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training, momentum=0.9):
        """批量归一化前向传播"""
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            x_hat = (x - mu) / np.sqrt(var + 1e-8)
            out = gamma * x_hat + beta
            running_mean = momentum * running_mean + (1 - momentum) * mu
            running_var = momentum * running_var + (1 - momentum) * var
            return out, x_hat, mu, var
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + 1e-8)
            out = gamma * x_hat + beta
            return out, x_hat, None, None  # 测试模式下返回 None

    def forward(self, X, training=True):
        """前向传播"""
        self.cache = []
        h = X

        for i, layer in enumerate(self.layers[:-1]):
            z = h @ layer['W'] + layer['b']
            if self.use_bn:
                gamma, beta = layer['gamma'], layer['beta']
                running_mean, running_var = layer['running_mean'], layer['running_var']
                z, x_hat, mu, var = self.batch_norm_forward(z, gamma, beta, running_mean, running_var, training)
                if training:
                    self.cache.append({'z': z, 'x_hat': x_hat, 'mu': mu, 'var': var, 'h': h})
                else:
                    self.cache.append({'z': z, 'x_hat': x_hat, 'h': h})  # 测试模式下不保存 mu 和 var
            else:
                self.cache.append({'z': z, 'h': h})
            h = self.relu(z)
            if training and self.dropout_prob > 0:
                mask = (np.random.rand(*h.shape) > self.dropout_prob) / (1 - self.dropout_prob)
                h *= mask
                self.cache[-1]['mask'] = mask

        z = h @ self.layers[-1]['W'] + self.layers[-1]['b']
        probs = self.softmax(z)
        self.cache.append({'z': z, 'probs': probs, 'h': h})
        return probs

    def compute_loss(self, probs, y):
        """计算交叉熵损失"""
        n = y.shape[0]
        log_probs = -np.log(probs[range(n), y.argmax(axis=1)] + 1e-8)
        return np.sum(log_probs) / n

    def backward(self, X, y):
        """反向传播"""
        n = X.shape[0]
        grads = [{} for _ in self.layers]
        probs = self.cache[-1]['probs']
        dz = probs - y
        dh = dz @ self.layers[-1]['W'].T
        grads[-1]['W'] = self.cache[-1]['h'].T @ dz / n
        grads[-1]['b'] = np.sum(dz, axis=0, keepdims=True) / n

        for i in range(len(self.layers) - 2, -1, -1):
            dz = dh * self.relu_grad(self.cache[i]['z'])
            if self.use_bn:
                x_hat, mu, var = self.cache[i]['x_hat'], self.cache[i]['mu'], self.cache[i]['var']
                gamma = self.layers[i]['gamma']
                dgamma = np.sum(dz * x_hat, axis=0, keepdims=True) / n
                dbeta = np.sum(dz, axis=0, keepdims=True) / n
                dx_hat = dz * gamma
                dvar = np.sum(dx_hat * (self.cache[i]['z'] - mu) * -0.5 * (var + 1e-8)**(-1.5), axis=0, keepdims=True)
                dmu = np.sum(dx_hat * -1 / np.sqrt(var + 1e-8), axis=0, keepdims=True) + dvar * -2 * np.sum(self.cache[i]['z'] - mu, axis=0, keepdims=True) / n
                dz = dx_hat / np.sqrt(var + 1e-8) + dvar * 2 * (self.cache[i]['z'] - mu) / n + dmu / n
                grads[i]['gamma'] = dgamma
                grads[i]['beta'] = dbeta
            if 'mask' in self.cache[i]:
                dz *= self.cache[i]['mask']
            h = self.cache[i]['h']
            grads[i]['W'] = h.T @ dz / n
            grads[i]['b'] = np.sum(dz, axis=0, keepdims=True) / n
            dh = dz @ self.layers[i]['W'].T

        return grads

    def update(self, grads, lr, momentum, weight_decay):
        """更新参数"""
        for i, layer in enumerate(self.layers):
            grads[i]['W'] += weight_decay * layer['W']
            self.velocities[i]['W'] = momentum * self.velocities[i]['W'] - lr * grads[i]['W']
            self.velocities[i]['b'] = momentum * self.velocities[i]['b'] - lr * grads[i]['b']
            layer['W'] += self.velocities[i]['W']
            layer['b'] += self.velocities[i]['b']
            if self.use_bn and i < len(self.layers) - 1:
                self.velocities[i]['gamma'] = momentum * self.velocities[i]['gamma'] - lr * grads[i]['gamma']
                self.velocities[i]['beta'] = momentum * self.velocities[i]['beta'] - lr * grads[i]['beta']
                layer['gamma'] += self.velocities[i]['gamma']
                layer['beta'] += self.velocities[i]['beta']