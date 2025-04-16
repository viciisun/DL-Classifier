import numpy as np

def load_and_preprocess_data(train_data_path='data/train_data.npy', train_label_path='data/train_label.npy',
                             test_data_path='data/test_data.npy', test_label_path='data/test_label.npy'):


    X_train = np.load(train_data_path)
    y_train = np.load(train_label_path).flatten()
    X_test = np.load(test_data_path)
    y_test = np.load(test_label_path).flatten()

    # Z-score normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8 
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # One-hot encoding
    def one_hot(labels, num_classes):
        return np.eye(num_classes)[labels]

    y_train_onehot = one_hot(y_train, 10)
    y_test_onehot = one_hot(y_test, 10)

    return X_train, y_train_onehot, X_test, y_test_onehot, y_test