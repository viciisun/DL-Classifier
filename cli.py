import argparse
from datetime import datetime

class CaseInsensitiveArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        # Convert all argument strings to lowercase
        if args is not None:
            args = [arg.lower() if not arg.startswith('--') else arg for arg in args]
        return super().parse_args(args, namespace)

def parse_arguments():
    """Parse command line arguments"""
    parser = CaseInsensitiveArgumentParser(
        description='Neural Network Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 main.py
  python3 main.py --preprocess standard
  python3 main.py --preprocess minmax

  # Using different features
  python3 main.py --gelu --adam
  python3 main.py --preprocess minmax --gelu
  python3 main.py --dropout 0.3 --gelu

  # With custom hyperparameters
  python3 main.py --gelu --adam --hidden_sizes 512 256 --lr 0.001 --dropout 0.5

  # Compare models without training
  python3 main.py --compare_models

Available Options:
  Preprocessing:
    --preprocess standard : Use standard scaling (mean=0, std=1)
    --preprocess minmax  : Use min-max scaling (range [0,1])
    --preprocess none    : No preprocessing (default)
  
  Features:
    --gelu  : Use GELU activation instead of ReLU
    --adam  : Use Adam optimizer instead of SGD
        """)
    
    # Preprocessing method
    parser.add_argument('--preprocess', type=str.lower, choices=['standard', 'minmax', 'none'], default='none',
                      help='Preprocessing method to use (default: none)')
    
    # Model comparison
    parser.add_argument('--compare_models', action='store_true',
                      help='Compare models without training')
    
    # Feature flags (simplified names)
    feature_group = parser.add_argument_group('Features')
    feature_group.add_argument('--gelu', action='store_true',
                           help='Use GELU activation instead of ReLU')
    feature_group.add_argument('--adam', action='store_true',
                           help='Use Adam optimizer instead of SGD')
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 128],
                        help='List of hidden layer sizes (default: [256, 128])')
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    training_group.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size (default: 128)')
    training_group.add_argument('--lr', type=float,
                        help='Learning rate (default: 1e-3 for Adam, 1e-2 for SGD)')
    training_group.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum coefficient for SGD (default: 0.9)')
    training_group.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) coefficient (default: 1e-4)')
    training_group.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (default: 0.5)')
    training_group.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping (default: 10)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')
    output_group.add_argument('--model_name', type=str, default=None,
                        help='Custom model name (default: auto-generated based on features)')
    
    args = parser.parse_args()
    
    # Set default learning rate based on optimizer
    if args.lr is None:
        args.lr = 1e-3 if args.adam else 1e-2
    
    # Validate numeric parameters
    if args.epochs <= 0:
        parser.error("Number of epochs must be positive")
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    if args.lr <= 0:
        parser.error("Learning rate must be positive")
    if not 0 <= args.momentum < 1:
        parser.error("Momentum must be between 0 and 1")
    if args.weight_decay < 0:
        parser.error("Weight decay must be non-negative")
    if not 0 <= args.dropout < 1:
        parser.error("Dropout probability must be between 0 and 1")
    if args.early_stopping_patience <= 0:
        parser.error("Early stopping patience must be positive")
    if not args.hidden_sizes:
        parser.error("Must specify at least one hidden layer size")
    
    # Convert feature flags to list of features for compatibility
    args.features = []
    if args.gelu:
        args.features.append('GELU')
    if args.adam:
        args.features.append('Adam')
    
    # Store preprocess_method for compatibility
    args.preprocess_method = args.preprocess
    
    return args

def generate_model_name(args):
    """Generate model name based on preprocessing method and features"""
    features = [args.preprocess_method] + args.features
    if args.model_name is None:
        return "_".join(sorted(features))
    return args.model_name 