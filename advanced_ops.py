import numpy as np

def gelu(x):
    """
    GELU activation function: x * Φ(x)
    where Φ(x) is the standard Gaussian CDF
    
    This is a smoother version of ReLU that has shown good performance in
    transformer architectures.
    
    Reference: Gaussian Error Linear Units (GELUs) - https://arxiv.org/abs/1606.08415
    
    Args:
        x: Input tensor
        
    Returns:
        GELU activation applied to input
    """
    # Approximation of GELU
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x):
    """
    Gradient of GELU activation function
    
    Args:
        x: Input tensor
        
    Returns:
        Gradient of GELU at x
    """
    # Approximation of GELU gradient
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    pdf = 0.5 * np.sqrt(2.0 / np.pi) * (1.0 + 0.134145 * x**2) * (1 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))**2)
    return cdf + x * pdf

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam Optimizer implementation
        
        Args:
            learning_rate (float): Learning rate
            beta1 (float): Exponential decay rate for first moment estimates
            beta2 (float): Exponential decay rate for second moment estimates
            epsilon (float): Small constant for numerical stability
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

class LRScheduler:
    def __init__(self, initial_lr=0.001, scheduler_type='step', **kwargs):
        """
        Learning Rate Scheduler
        
        Args:
            initial_lr (float): Initial learning rate
            scheduler_type (str): Type of scheduler ('step', 'cosine', 'exponential')
            **kwargs: Additional arguments for specific schedulers
                - step_size: Number of epochs between lr decay (for 'step')
                - gamma: Multiplicative factor for lr decay (for 'step' and 'exponential')
                - total_epochs: Total number of epochs (for 'cosine')
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
    
    def step(self, epoch):
        """
        Update learning rate based on current epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Updated learning rate
        """
        if self.scheduler_type == 'step':
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            self.current_lr = self.initial_lr * gamma ** (epoch // step_size)
        
        elif self.scheduler_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            self.current_lr = self.initial_lr * gamma ** epoch
        
        elif self.scheduler_type == 'cosine':
            total_epochs = self.kwargs.get('total_epochs', 100)
            self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        
        return self.current_lr 