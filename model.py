import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=128, hidden_sizes=[256, 128], output_size=10, 
                 use_bn=True, dropout_prob=0.5, activation='relu'):
        """
        Initialize the neural network
        
        Args:
            input_size (int): Input feature dimension
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Number of output classes
            use_bn (bool): Whether to use batch normalization
            dropout_prob (float): Dropout probability (0-1)
            activation (str): Activation function ('relu' or 'gelu')
        """
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        self.use_bn = use_bn
        self.dropout_prob = dropout_prob
        self.activation = activation

        # Xavier initialization for weights and biases
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.layers.append({'W': w, 'b': b})
        
        # Batch normalization parameters
        if use_bn:
            for i in range(len(hidden_sizes)):
                self.layers[i]['gamma'] = np.ones((1, sizes[i+1]))
                self.layers[i]['beta'] = np.zeros((1, sizes[i+1]))
                self.layers[i]['running_mean'] = np.zeros((1, sizes[i+1]))
                self.layers[i]['running_var'] = np.ones((1, sizes[i+1]))

        # Momentum SGD velocities
        self.velocities = [{'W': np.zeros_like(l['W']), 'b': np.zeros_like(l['b'])} 
                          for l in self.layers]
        if use_bn:
            for i in range(len(hidden_sizes)):
                self.velocities[i]['gamma'] = np.zeros_like(self.layers[i]['gamma'])
                self.velocities[i]['beta'] = np.zeros_like(self.layers[i]['beta'])

    def activate(self, x):
        """
        Apply activation function
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'gelu':
            return gelu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def activate_grad(self, x):
        """
        Calculate gradient of activation function
        
        Args:
            x: Input tensor
            
        Returns:
            Gradient of activation function
        """
        if self.activation == 'relu':
            return self.relu_grad(x)
        elif self.activation == 'gelu':
            return gelu_grad(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_grad(self, x):
        """ReLU gradient"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training, momentum=0.9):
        """
        Batch normalization forward pass
        
        Args:
            x: Input tensor
            gamma: Scale parameter
            beta: Shift parameter
            running_mean: Running mean for inference
            running_var: Running variance for inference
            training: Whether in training mode
            momentum: Momentum for running statistics
            
        Returns:
            Normalized tensor and cache for backprop
        """
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
            return out, x_hat, None, None  # None for test mode

    def forward(self, X, training=True):
        """
        Forward pass through the network
        
        Args:
            X: Input features
            training: Whether in training mode
            
        Returns:
            Output probabilities
        """
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
                    self.cache.append({'z': z, 'x_hat': x_hat, 'h': h})
            else:
                self.cache.append({'z': z, 'h': h})
            
            h = self.activate(z)
            
            if training and self.dropout_prob > 0:
                mask = (np.random.rand(*h.shape) > self.dropout_prob) / (1 - self.dropout_prob)
                h *= mask
                self.cache[-1]['mask'] = mask

        z = h @ self.layers[-1]['W'] + self.layers[-1]['b']
        probs = self.softmax(z)
        self.cache.append({'z': z, 'probs': probs, 'h': h})
        return probs

    def compute_loss(self, probs, y):
        """
        Compute cross-entropy loss
        
        Args:
            probs: Predicted probabilities
            y: True labels (one-hot encoded)
            
        Returns:
            Cross-entropy loss
        """
        n = y.shape[0]
        log_probs = -np.log(probs[range(n), y.argmax(axis=1)] + 1e-8)
        return np.sum(log_probs) / n

    def backward(self, X, y):
        """
        Backward pass through the network
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Gradients for each parameter
        """
        n = X.shape[0]
        grads = [{} for _ in self.layers]
        probs = self.cache[-1]['probs']
        dz = probs - y
        dh = dz @ self.layers[-1]['W'].T
        grads[-1]['W'] = self.cache[-1]['h'].T @ dz / n
        grads[-1]['b'] = np.sum(dz, axis=0, keepdims=True) / n

        for i in range(len(self.layers) - 2, -1, -1):
            dz = dh * self.activate_grad(self.cache[i]['z'])
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
        """
        Update parameters using SGD with momentum
        
        Args:
            grads: Gradients for each parameter
            lr: Learning rate
            momentum: Momentum coefficient
            weight_decay: Weight decay (L2 regularization) coefficient
        """
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

class AdamOptimizer:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam Optimizer implementation
        
        Args:
            learning_rate: Learning rate (default: 1e-3)
            beta1: Exponential decay rate for first moment estimates (default: 0.9)
            beta2: Exponential decay rate for second moment estimates (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimates
        self.v = None  # Second moment estimates
        self.t = 0     # Timestep
    
    def initialize(self, layers):
        """
        Initialize moment estimates for each parameter
        
        Args:
            layers: List of layers containing weights and biases
        """
        self.m = []
        self.v = []
        for layer in layers:
            m_layer = {}
            v_layer = {}
            for param_name, param in layer.items():
                if isinstance(param, np.ndarray):
                    m_layer[param_name] = np.zeros_like(param)
                    v_layer[param_name] = np.zeros_like(param)
            self.m.append(m_layer)
            self.v.append(v_layer)
    
    def update(self, layers, grads):
        """
        Update parameters using Adam optimizer
        
        Args:
            layers: List of layers containing weights and biases
            grads: List of gradients for each parameter
        """
        if self.m is None:
            self.initialize(layers)
        
        self.t += 1
        
        for i, (layer, grad) in enumerate(zip(layers, grads)):
            for param_name, g in grad.items():
                if isinstance(g, np.ndarray):
                    self.m[i][param_name] = self.beta1 * self.m[i][param_name] + (1 - self.beta1) * g
                    self.v[i][param_name] = self.beta2 * self.v[i][param_name] + (1 - self.beta2) * g**2
                    
                    # Bias correction
                    m_corrected = self.m[i][param_name] / (1 - self.beta1**self.t)
                    v_corrected = self.v[i][param_name] / (1 - self.beta2**self.t)
                    
                    # Update parameters
                    layer[param_name] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

def gelu(x):
    """
    GELU activation function: x * Φ(x)
    where Φ(x) is the standard Gaussian CDF
    
    Args:
        x: Input tensor
    Returns:
        GELU activation applied to input
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x):
    """
    Gradient of GELU activation function
    
    Args:
        x: Input tensor
    Returns:
        Gradient of GELU at x
    """
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    pdf = 0.5 * np.sqrt(2.0 / np.pi) * (1.0 + 0.134145 * x**2) * (1 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))**2)
    return cdf + x * pdf
