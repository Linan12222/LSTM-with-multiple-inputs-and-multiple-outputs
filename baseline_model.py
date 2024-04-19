import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from tqdm import tqdm


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
        self.terminal.flush()
        self.log.flush()

sys.stdout = StreamToLogger("console_output_GRU.txt")

def negative_log_likelihood(y_pred, y_true, sigma):
    mu, sigma = y_pred
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y_true).mean()

def calculate_intervals(mu, sigma, coverage=0.80):
    z = norm.ppf(1 - (1 - coverage) / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return lower, upper

def PICP(y_true, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)

def ACIW(lower, upper):
    return np.mean(upper - lower)

def PINAW(y_true, lower, upper):
    data_range = y_true.max() - y_true.min()
    interval_width = upper - lower
    return np.mean(interval_width) / data_range

def CWC(PICP_value, PINAW_value, desired_coverage=0.85):
    k = 10
    PICP_penalty = max(0, (desired_coverage - PICP_value))
    return PINAW_value * (1 + k * PICP_penalty)

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_out_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_out_len = seq_out_len
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, output_dim * seq_out_len)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim * seq_out_len)
    # rnn and gru
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out))
        return mu.view(x.size(0), self.seq_out_len, self.output_dim), sigma.view(x.size(0), self.seq_out_len, self.output_dim)

    # #lstm
    # def forward(self, x):
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
    #     c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始化细胞状态
    #     out, _ = self.gru(x, (h0, c0))  # 同时传递隐藏状态和细胞状态
    #     out = out[:, -1, :]
    #     mu = self.fc_mu(out)
    #     sigma = torch.exp(self.fc_sigma(out))
    #     return mu.view(x.size(0), self.seq_out_len, self.output_dim), sigma.view(x.size(0), self.seq_out_len,
    #                                                                              self.output_dim)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel('17+18-已处理.xlsx')
data = data.iloc[1:, 1:]
look_back = 4

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

seq_out_lens = [1, 4, 8, 12, 16]
picp_scores, aciw_scores, pinaw_scores, cwc_scores = [], [], [], []

# 对每个序列长度进行训练和评估
for seq_out_len in seq_out_lens:
    picp_vals, aciw_vals, pinaw_vals, cwc_vals = [], [], [], []

    for run in range(10):  # 训练模型十次
        print(f"Training for seq_out_len = {seq_out_len}, Run {run + 1}/10")
        x, y = split_sequence(data, look_back, seq_out_len)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

        model = Model(input_dim=X_train.shape[-1], hidden_dim=64, num_layers=1, output_dim=14, seq_out_len=seq_out_len)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 训练过程
        for epoch in tqdm(range(50), desc="Epochs"):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                mu, sigma = model(inputs)
                loss = negative_log_likelihood((mu, sigma), targets, sigma)
                loss.backward()
                optimizer.step()

        # 评估过程
        model.eval()
        with torch.no_grad():
            mu, sigma = model(torch.from_numpy(X_test).float().to(device))
            mu = mu.cpu().detach().numpy()
            sigma = sigma.cpu().detach().numpy()
            lower, upper = calculate_intervals(mu, sigma)
            picp = PICP(y_test, lower, upper)
            aciw = ACIW(lower, upper)
            pinaw = PINAW(y_test, lower, upper)
            cwc = CWC(picp, pinaw)

            picp_vals.append(picp)
            aciw_vals.append(aciw)
            pinaw_vals.append(pinaw)
            cwc_vals.append(cwc)

            # 输出当前运行的评估指标
            print(f"Run {run + 1} - PICP: {picp}, ACIW: {aciw}, PINAW: {pinaw}, CWC: {cwc}")

    # 计算平均评估指标
    avg_picp = np.mean(picp_vals)
    avg_aciw = np.mean(aciw_vals)
    avg_pinaw = np.mean(pinaw_vals)
    avg_cwc = np.mean(cwc_vals)

    # 输出当前seq_out_len的平均评估指标
    print(
        f'Average for seq_out_len = {seq_out_len}: PICP: {avg_picp}, ACIW: {avg_aciw}, PINAW: {avg_pinaw}, CWC: {avg_cwc}')
    print('-' * 50)
