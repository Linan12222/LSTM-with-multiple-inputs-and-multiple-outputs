import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

class StreamToLogger(object):
    """
    自定义流，用于同时将输出信息发送到标准输出和文件。
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这个函数在这里是为了兼容文件对象的接口
        self.terminal.flush()
        self.log.flush()

sys.stdout = StreamToLogger("console_output_RNN.txt")

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_out_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim  # 添加这一行来初始化output_dim
        self.seq_out_len = seq_out_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * seq_out_len)  # 输出层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, h0)  # 只传递 h0
        out = out[:, -1, :]  # 只取 RNN 最后一个时间步的输出
        out = self.fc(out)
        return out.view(x.size(0), -1, self.output_dim)  # 重塑输出以匹配预期的多步格式


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

data = pd.read_excel('风机故障数据-已处理.xlsx', skiprows=1)
# 删除最后一行
data = data.iloc[:-1, :-1]
N = data.shape[1]  # 获取数据集的总列数
data= data.iloc[:, 13:N]

look_back = 6

# 窗口函数
# 修改split_sequence函数来生成多步标签
def split_sequence(sequence, look_back, seq_out_len):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + look_back
        out_end_ix = end_ix + seq_out_len
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence.iloc[i:end_ix, :].values, sequence.iloc[end_ix:out_end_ix, :].values
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)




# 初始数据读取和处理
x, y = split_sequence(data, look_back=look_back, seq_out_len=1)  # 初始值仅用于获取数据维度
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 设置循环
seq_out_lens = [1, 4, 8, 13, 16]

for seq_out_len in seq_out_lens:
    print('seq_out_len:', seq_out_len)
    x, y = split_sequence(data, look_back=look_back, seq_out_len=seq_out_len)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 数据转换
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    epoches = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 初始化指标存储列表
    mse_scores, rmse_scores, mae_scores, r2_scores, mape_scores = [], [], [], [], []

    # 运行模型 10 次
    num_runs = 10
    for run in range(num_runs):
        # 模型定义
        model = Model(input_dim=X_train.shape[-1], hidden_dim=64, num_layers=1, output_dim=29, seq_out_len=seq_out_len)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 模型训练
        model.to(device)
        for epoch in range(epoches):
            model.train()
            train_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # 模型测试
        model.eval()
        preds = model(X_test.to(device))
        predictions = preds.view(-1, seq_out_len, y_train.shape[-1]).cpu().detach().numpy()
        predictions_reshaped = predictions.reshape(-1, y_train.shape[-1])
        y_test_reshaped = y_test.numpy().reshape(-1, y_train.shape[-1])

        # 计算评估指标
        mse = mean_squared_error(y_test_reshaped, predictions_reshaped)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reshaped, predictions_reshaped)
        r2 = r2_score(y_test_reshaped, predictions_reshaped)
        mape = np.mean(np.abs((y_test_reshaped - predictions_reshaped) / y_test_reshaped)) * 100

        # 存储指标
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        mape_scores.append(mape)

        # 打印当前迭代的评估指标
        print(f'Run {run + 1}:')
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('MAE:', mae)
        print('R-squared:', r2)
        print('MAPE:', mape)
        print('-' * 50)

    # 计算平均评估指标
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_mape = np.mean(mape_scores)


    # 打印平均评估指标
    print('Average Metrics Over 10 Runs for seq_out_len =', seq_out_len)
    print('Average MSE:', avg_mse)
    print('Average RMSE:', avg_rmse)
    print('Average MAE:', avg_mae)
    print('Average R-squared:', avg_r2)
    print('Average MAPE:', avg_mape)
    print('-' * 100)
