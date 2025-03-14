import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):  # num_classes 默认为 2（二分类）
        super(MLPClassifier, self).__init__()
        # 将input_dim/2转换为整数
        hidden_dim = int(input_dim / 2)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 加入 BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 加入 BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),  # 输出层
        )

        self.apply(self.init_weights)  # 应用权重初始化

    def forward(self, x):
        # 直接返回logits，不应用激活函数
        # CrossEntropyLoss会在内部应用Softmax
        return self.model(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # He 初始化
            if m.bias is not None:
                m.bias.data.zero_()


def evaluate_accuracy(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        # 多分类任务：取最大概率的类别作为预测结果
        preds = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())
    return accuracy


# 训练模型（带早停策略）
def train_model(X, y, k_folds=5, epochs=1000, lr=0.001, batch_size=64, patience=50):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_losses = {"train": [], "val": []}
    best_scores = []  # 记录每折的最佳验证损失
    best_models = []  # 保存每折的最佳模型

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold+1}/{k_folds}")

        # 数据划分
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 转换 Tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        # 多分类任务，目标变量应为整数张量
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        model = MLPClassifier(X_train.shape[1]).to(device)

        # 多分类任务使用CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )  # weight_decay=1e-4 可防止权重过大导致过拟合
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        # 记录损失
        fold_train_loss = []
        fold_val_loss = []
        best_val_loss = float("inf")
        patience_counter = 0  # 早停计数器
        best_model_state = None  # 保存最佳模型的状态字典

        # 训练循环
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)

            # 计算平均损失
            epoch_train_loss /= len(train_loader.dataset)
            fold_train_loss.append(epoch_train_loss)

            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                fold_val_loss.append(val_loss)

            # 学习率调整
            scheduler.step(val_loss)

            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # 重新计数
                best_model_state = model.state_dict()  # 保存最佳模型状态
            else:
                patience_counter += 1

            # 打印日志
            if (epoch + 1) % 20 == 0:
                val_accuracy = evaluate_accuracy(model, X_val_tensor, y_val_tensor)
                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
                )

            # 触发早停
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} | Best Val Loss: {best_val_loss:.4f}"
                )
                break

        # 加载最佳模型
        best_model = MLPClassifier(X_train.shape[1]).to(device)
        best_model.load_state_dict(best_model_state)
        best_models.append(best_model)  # 保存最佳模型

        # 记录当前 fold 的损失
        fold_losses["train"].append(fold_train_loss)
        fold_losses["val"].append(fold_val_loss)
        best_scores.append(best_val_loss)  # 记录最佳验证损失

    # 计算所有折的平均最佳验证损失
    final_score = np.mean(best_scores)
    print(f"\nFinal Model Score (Avg Best Val Loss): {final_score:.4f}")

    # 返回所有折的最佳模型和平均最佳验证损失
    return best_models, fold_losses, final_score


# 使用示例
def example_usage():
    # 假设X_train和y_train已经准备好
    # X_train, X_valid, y_train, y_valid = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2, random_state=42)

    # 训练模型
    # best_models, fold_losses, final_score = train_model(X_train.values.astype(float), y_train.values.astype(int),
    #                                       k_folds=5, epochs=1000, lr=0.001, batch_size=64, patience=50)

    # 可视化
    # plt.figure(figsize=(10, 6))
    # for fold in range(5):
    #     plt.plot(fold_losses['train'][fold], label=f'Fold {fold+1} Train', alpha=0.5)
    #     plt.plot(fold_losses['val'][fold], label=f'Fold {fold+1} Val', linestyle='--', alpha=0.5)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss per Fold')
    # plt.legend()
    # plt.show()

    # 使用最佳模型进行预测
    # best_model = best_models[-1]  # 选择最后一个折的最佳模型
    # X_valid_tensor = torch.tensor(X_valid.values.astype(float), dtype=torch.float32).to(device)
    # y_valid_tensor = torch.tensor(y_valid.values.astype(int), dtype=torch.long).to(device)
    # test_accuracy = evaluate_accuracy(best_model, X_valid_tensor, y_valid_tensor)
    # print(f"Test Accuracy of Best Model: {test_accuracy:.4f}")
    pass


if __name__ == "__main__":
    example_usage()
