from data_loader import load_and_preprocess_data
from model import NeuralNetwork
from train import train_model
from evaluate import evaluate_model
from visualize import plot_confusion_matrix

if __name__ == "__main__":
    # 加载数据
    X_train_full, y_train_onehot, X_test, y_test_onehot, y_test = load_and_preprocess_data()

    # 分割训练集和验证集（80% 训练，20% 验证）
    n_train = int(0.8 * X_train_full.shape[0])
    X_train = X_train_full[:n_train]
    y_train = y_train_onehot[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_onehot[n_train:]

    # 初始化模型
    model = NeuralNetwork(input_size=128, hidden_sizes=[256, 128], output_size=10, 
                          use_bn=True, dropout_prob=0.5)

    # 训练模型
    best_model_params = train_model(model, X_train, y_train, X_val, y_val, 
                                    epochs=100, batch_size=128, lr=0.001, 
                                    momentum=0.9, weight_decay=1e-4)

    # 加载最佳模型
    model.layers = best_model_params

    # 评估测试集
    test_acc, cm = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(cm)