from tqdm import tqdm
import numpy as np

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, 
                lr=0.001, momentum=0.9, weight_decay=1e-4):
    """训练模型"""
    n = X_train.shape[0]
    best_val_acc = 0.0
    best_model_params = None

    for epoch in range(epochs):
        # 打乱训练数据
        indices = np.random.permutation(n)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # 小批量训练
        with tqdm(total=n // batch_size, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(0, n, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                probs = model.forward(X_batch, training=True)
                loss = model.compute_loss(probs, y_batch)
                grads = model.backward(X_batch, y_batch)
                model.update(grads, lr, momentum, weight_decay)
                pbar.set_postfix(loss=loss)
                pbar.update(1)

        # 验证集评估
        val_probs = model.forward(X_val, training=False)
        val_loss = model.compute_loss(val_probs, y_val)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == np.argmax(y_val, axis=1))
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = [{k: v.copy() for k, v in layer.items()} for layer in model.layers]

    return best_model_params