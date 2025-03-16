from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    probs = model.forward(X_test, training=False)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_test)
    cm = confusion_matrix(y_test, preds)
    return acc, cm