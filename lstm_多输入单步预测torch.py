import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Subset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# data_ = pd.read_excel('./2018风电数据_modify.xlsx')
data_ = pd.read_excel('./data_proccessed.xlsx')
look_back = 4

# 归一化
scaler = MinMaxScaler((0, 1))
data_scale = scaler.fit_transform(data_)

# 窗口函数
def split_sequence(sequence, look_back):
    X, y = [], []
    for i in range(len(sequence)):
        end_element_index = i + look_back
        if end_element_index > len(sequence) - 1:  # 序列中最后一个元素的索引
            break
        sequence_x, sequence_y = sequence[i:end_element_index], sequence[end_element_index]  # 取最后一个元素作为预测值y
        X.append(sequence_x)
        if len(sequence_y.shape) > 1:
            y.append(sequence_y)
        else:
            y.append(sequence_y)

    return np.array(X), np.array(y)

# 划分数据集
x, y = split_sequence(data_scale, look_back=look_back)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# 转换为PyTorch的Tensor
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# 再将训练集划分为训练集和验证集
train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_size = int(len(train_ds) * 0.1)
train_ds, val_ds = Subset(train_ds, range(len(train_ds) - val_size)), Subset(train_ds, range(len(train_ds)-val_size, len(train_ds)))
train_loader = DataLoader(train_ds, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1000, shuffle=False)

# 模型建立
from model import LSTMModel
model = LSTMModel(input_dim=X_train.shape[-1], hidden_dim=64, num_layers=1, output_dim=y_train.shape[-1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模型训练和验证
epo_losses = []
epoches = 60
best_loss = np.inf
for epoch in range(epoches):
    train_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    with torch.no_grad():
        val_loss = 0
        for j, (inputs, targets) in enumerate(val_loader):
            val_output = model(inputs)
            loss = criterion(val_output, targets)
            val_loss += loss.item()
        
    val_loss /= (j + 1)
    train_loss /= (i + 1)
    epo_losses.append(train_loss)

    # 模型保存
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        
    print(f"Epoche {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


# ---------------------------模型测试----------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
preds = model(X_test.to(device))
predictions = scaler.inverse_transform(preds.cpu().detach().numpy())

# 预测结果
df_pre = pd.DataFrame(predictions)
# 保存预测结果到excel
df_pre.to_excel('./predictions.xlsx')
# 真实值
df_real = pd.DataFrame(scaler.inverse_transform(y_test.numpy()))

# ----------------------------------指标计算------------------------#
mae = (df_pre - df_real).abs().mean()
mse = ((df_pre - df_real) ** 2).mean()
r2 = df_pre.apply(lambda col: r2_score(col, df_real[col.name]))
# 每一列的MAE、MSE、R2
print('MAE:\n', mae)
print('MSE:\n', mse)
print('R2:\n', r2)

# 画图
for col in df_real.columns:
    plt.figure(figsize=(30, 8), dpi=100)
    plt.plot(df_real[col], label='df_real', marker='.')
    plt.plot(df_pre[col], label='df_pre', marker='*')
    plt.legend()
    plt.title(col)
    plt.show()