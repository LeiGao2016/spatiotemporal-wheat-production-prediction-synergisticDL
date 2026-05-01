import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):


    def __init__(self, hidden_size, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, lstm_output):

        attn_output, _ = self.attn(lstm_output, lstm_output, lstm_output)
        return attn_output.mean(dim=1)  # 取序列维度的均值

class LSTMWithMultiHeadAttentionModel(nn.Module):


    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=4):
        super(LSTMWithMultiHeadAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 注意：双向LSTM的输出维度是 hidden_size * 2
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        out = self.fc(attn_out)
        return out

class LSTMModel(nn.Module): # 简化版 LSTM，用于对比
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# --- 静态模型 (1D-CNN) ---

class StaticCNNModel(nn.Module):

    def __init__(self, input_channels, hidden_size, kernel_size=3, output_size=1):
        super(StaticCNNModel, self).__init__()


        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(32) # BatchNormalization 帮助稳定训练
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=kernel_size, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()

        # 全连接层，将卷积后的特征转换为最终输出
        # 如果使用AdaptiveAvgPool1d，其输出维度是 hidden_size
        # 如果没有使用池化，需要知道卷积后的序列长度，然后乘以输出通道数
        # 这里我们假设经过多层卷积后，我们希望得到一个 hidden_size 维度的表示
        # 然后通过一个线性层映射到 output_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # 期望输入 x 的形状是 (batch_size, num_static_features)
        # 1D CNN 需要 (batch_size, channels, seq_len)
        # 我们将 num_static_features 作为 channels，seq_len 设置为 1
        x = x.unsqueeze(2) # 增加一个序列长度维度，形状变为 (batch_size, num_static_features, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 在seq_len=1的情况下，我们需要对卷积后的输出进行处理
        # AdaptiveAvgPool1d 可以将任意 seq_len 的输出池化到固定长度，这里池化到 1
        x = F.adaptive_avg_pool1d(x, 1) # 输出形状 (batch_size, hidden_size, 1)

        # 展平张量，准备送入全连接层
        x = x.view(x.size(0), -1) # 输出形状 (batch_size, hidden_size)

        out = self.fc(x)
        return out

# --- 融合模型 ---

class IntegratedModel(nn.Module):

    def __init__(self, dynamic_model, static_model, fused_hidden_size, output_size=1):
        super(IntegratedModel, self).__init__()
        self.dynamic_model = dynamic_model
        self.static_model = static_model


        self.fc_fuse = nn.Linear(fused_hidden_size, output_size)

    def forward(self, dynamic_input, static_input):

        # 获取动态模型和静态模型的输出
        dynamic_output = self.dynamic_model(dynamic_input)
        static_output = self.static_model(static_input)

        # 拼接两个模型的输出
        # dynamic_output 形状：(batch_size, dynamic_output_dim)
        # static_output 形状：(batch_size, static_output_dim)
        # 假设 dynamic_output_dim 和 static_output_dim 都是 output_size (例如 1)
        # 那么拼接后是 (batch_size, dynamic_output_dim + static_output_dim)
        fused_features = torch.cat((dynamic_output, static_output), dim=1)

        # 通过一个线性层进行融合和变换
        out = self.fc_fuse(fused_features)
        return out

# --- 数据加载与预处理 ---

class WheatYieldDataset(Dataset):
    """
    自定义数据集类，用于加载和批处理小麦产量数据。
    """
    def __init__(self, dynamic_file_path: str, static_file_path: str,
                 dynamic_feature_cols: list, static_feature_cols: list, target_col: str,
                 window_size: int = 10, is_train: bool = True):

        self.dynamic_file_path = dynamic_file_path
        self.static_file_path = static_file_path
        self.dynamic_feature_cols = dynamic_feature_cols
        self.static_feature_cols = static_feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.is_train = is_train

        # 加载动态数据
        dynamic_df = pd.read_csv(dynamic_file_path)
        dynamic_data_all = dynamic_df[dynamic_feature_cols + [target_col]].values

        # 加载静态数据
        static_df = pd.read_csv(static_file_path)
        static_data_all = static_df[static_feature_cols].values


        if len(dynamic_data_all) != len(static_data_all):
            raise ValueError("Dynamic and static data have different lengths. Please ensure they are aligned.")

        self.dynamic_data = dynamic_data_all[:, :-1].astype(np.float32) # 动态特征
        self.static_data = static_data_all.astype(np.float32)       # 静态特征
        self.targets = dynamic_data_all[:, -1].astype(np.float32)    # 目标变量

        # --- 数据归一化 ---
        # 对动态特征和静态特征分别进行归一化，目标变量通常不需要归一化，或者可以单独处理
        self.dynamic_scaler = StandardScaler()
        self.dynamic_data = self.dynamic_scaler.fit_transform(self.dynamic_data)

        self.static_scaler = StandardScaler()
        self.static_data = self.static_scaler.fit_transform(self.static_data)

        # 目标变量也可能需要归一化，尤其是在使用 HuberLoss 或 MSELoss 时，数值范围过大会影响训练
        # self.target_scaler = StandardScaler()
        # self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).squeeze(-1)
        # TODO: 如果使用目标变量归一化，记得在预测时进行反归一化

        # --- 构建时间序列样本 ---
        self.dynamic_samples_x = []
        self.static_samples_x = []
        self.y_samples = []
        if self.is_train:
            for i in range(len(self.dynamic_data) - window_size + 1):
                # 动态数据样本
                dynamic_x_window = self.dynamic_data[i:i + window_size]
                self.dynamic_samples_x.append(dynamic_x_window)

                # 静态数据样本：对于每个动态时间窗口，我们使用一个对应的静态数据点
                # 假设静态数据样本的数量与最后一个动态时间步的索引对应
                self.static_samples_x.append(self.static_data[i + window_size -1]) # 取窗口最后一个时间步对应的静态特征

                # 目标变量，对应动态窗口的最后一个时间步的值
                self.y_samples.append(self.targets[i + window_size -1])
        else: # 测试集或验证集，可能不需要滑动窗口，直接使用最后一步预测
            # 如果是纯粹的预测，则直接获取最后一个 window_size 的数据
            self.dynamic_samples_x.append(self.dynamic_data[-window_size:])
            self.static_samples_x.append(self.static_data[-1]) # 假设最后一步对应最后一个静态特征
            # 目标变量在测试时是不知的，这里我们只是为了保持长度一致，实际训练时才是 y

        self.dynamic_samples_x = np.array(self.dynamic_samples_x)
        self.static_samples_x = np.array(self.static_samples_x)
        self.y_samples = np.array(self.y_samples)

        # 调整静态样本的形状以适应1D CNN (batch_size, num_static_features) -> (batch_size, num_static_features, 1)
        # 注意：这里静态模型 StaticCNNModel 的 init 中会做 unsqueeze(2)
        # 所以这里返回的静态样本形状是 (batch_size, num_static_features)

    def __len__(self):
        return len(self.y_samples)

    def __getitem__(self, idx):
        # 动态数据（用于 GLM-Attn）
        dynamic_x = torch.tensor(self.dynamic_samples_x[idx], dtype=torch.float32)
        # 静态数据（用于 1D-CNN）
        static_x = torch.tensor(self.static_samples_x[idx], dtype=torch.float32)
        # 目标变量
        y = torch.tensor(self.y_samples[idx], dtype=torch.float32)
        return dynamic_x, static_x, y

# --- 评估函数 (与之前相同) ---

def calculate_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    metrics = {'MSE': mse, 'RMSE': rmse, 'R²': r2}
    return metrics

# --- 训练与预测函数 ---

def train_integrated_model(
    dynamic_file_path: str,
    static_file_path: str,
    results_dir: str,
    dynamic_feature_cols: list,
    static_feature_cols: list,
    target_col: str,
    window_size: int = 10,
    epochs: int = 100,  # 增加训练轮数以提高性能
    batch_size: int = 32, # 增大 batch_size
    learning_rate: float = 0.0005, # 调整学习率
    dynamic_hidden_size: int = 64,
    dynamic_num_layers: int = 2,
    static_cnn_hidden_size: int = 32, # 1D CNN 的中间隐藏层大小
    static_cnn_kernel_size: int = 3
):

    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, 'integrated_training_log.txt')

    # 创建 Dataset 和 DataLoader
    # 训练集
    train_dataset = WheatYieldDataset(
        dynamic_file_path=dynamic_file_path,
        static_file_path=static_file_path,
        dynamic_feature_cols=dynamic_feature_cols,
        static_feature_cols=static_feature_cols,
        target_col=target_col,
        window_size=window_size,
        is_train=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证集（如果数据允许）
    # 为简化，这里暂时不严格划分验证集，直接在整个训练集上进行 eval，并保存最后 N 个的预测
    # 实际应用中，建议划分验证集来监控过拟合

    # 确定模型参数
    num_dynamic_features = len(dynamic_feature_cols)
    num_static_features = len(static_feature_cols)
    output_size = 1 # 小麦产量是单个值

    # --- 模型实例化 ---
    # 动态模型 (GLM-Attn)
    dynamic_model = LSTMWithMultiHeadAttentionModel(
        input_size=num_dynamic_features,
        hidden_size=dynamic_hidden_size,
        num_layers=dynamic_num_layers,
        output_size=output_size  # dynaimc model 的输出维度
    )

    # 静态模型 (1D-CNN)
    static_model = StaticCNNModel(
        input_channels=num_static_features,
        hidden_size=static_cnn_hidden_size,
        kernel_size=static_cnn_kernel_size,
        output_size=output_size # static model 的输出维度
    )

    # 融合模型
    # 拼接后的维度是 dynamic_model.output_size + static_model.output_size
    fused_input_dim = output_size + output_size
    integrated_model = IntegratedModel(
        dynamic_model=dynamic_model,
        static_model=static_model,
        fused_hidden_size=fused_input_dim,
        output_size=output_size
    )

    # --- 损失函数和优化器 ---
    criterion = nn.HuberLoss() # HuberLoss 对异常值鲁棒性更好
    optimizer = optim.Adam(integrated_model.parameters(), lr=learning_rate)

    # --- 训练 ---
    integrated_model.train()
    train_losses = []
    print(f"\n--- Starting training of Integrated Model ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch_dynamic_x, batch_static_x, batch_y in train_dataloader:
            optimizer.zero_grad()

            # 前向传播
            outputs = integrated_model(batch_dynamic_x, batch_static_x)
            loss = criterion(outputs.squeeze(), batch_y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

    # --- 评估与预测 ---
    integrated_model.eval()
    all_dynamic_x = torch.tensor(train_dataset.dynamic_samples_x, dtype=torch.float32)
    all_static_x = torch.tensor(train_dataset.static_samples_x, dtype=torch.float32)
    all_y = torch.tensor(train_dataset.y_samples, dtype=torch.float32)

    with torch.no_grad():
        all_predictions = integrated_model(all_dynamic_x, all_static_x).squeeze().numpy()

    # 假设我们只关心最后 N 个样本的预测，因为它们代表了最新的情况
    # 或者，我们可以根据你的需求，返回所有预测值
    num_last_samples =4  # 取最后5个或可用样本数

    true_values_eval = all_y[-num_last_samples:].numpy()
    predicted_values_eval = all_predictions[-num_last_samples:]

    metrics = calculate_metrics(true_values_eval, predicted_values_eval)

    print("\n--- Evaluation Metrics on Last {} Samples ---".format(num_last_samples))
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²: {metrics['R²']:.4f}")
    print("zuizhongjieguo:")
    # --- 保存结果 ---
    # 保存预测值
    prediction_df = pd.DataFrame({
        'True': true_values_eval,
        'Predicted': predicted_values_eval
    })
    save_csv_path = os.path.join(results_dir, os.path.basename(dynamic_file_path).replace('.csv', '_p.csv'))
    prediction_df.to_csv(save_csv_path, index=False)
    print(f"预测值已保存至 {save_csv_path}")

    return integrated_model, train_dataset  # 返回模型和数据集（用于后续可能的测试集加载）


# --- 主执行部分 ---

# 文件夹和文件配置
csv_dir = 'DATA_After'  # 动态气象数据目录
static_csv_dir = 'DATA_After'  # 静态土壤数据目录
results_dir = 'Results'

# 特征列定义
dynamic_feature_cols = ['Rainfall_Annual', 'Rainfall_Apr_Oct', 'Rainfall_Jun_Aug', 'Radiation_Annual', 'Radiation_Apr_Oct',
                        'Radiation_Jun_Aug', 'Tmax_Annual', 'Tmax_Apr_Oct', 'Tmax_Jun_Aug', 'Tmin_Annual', 'Tmin_Apr_Oct',
                        'Tmin_Jun_Aug', 'AET_Annual', 'AET_Apr_Oct', 'AET_Jun_Aug']
static_feature_cols = ['aridity_index', 'clay_depth_avg', 'silt_depth_avg', 'sand_depth_avg', 'regolith_depth', 'weathering_intensity_index'] # 假设的静态土壤特征
target_col = 'Yield_Grid'

# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)

# 遍历并训练模型
files=sorted(os.listdir(csv_dir),key=lambda x:int(''.join(filter(str.isdigit,x))))
for filename in files:
    if filename.endswith('.csv'):
        dynamic_file_path = os.path.join(csv_dir, filename)

        static_file_path = os.path.join(static_csv_dir, filename)  # 查找对应的静态数据文件

        if not os.path.exists(static_file_path):
            print(f"Warning: Static file not found for {filename} at {static_file_path}. Skipping.")
            continue

        print(f"\n--- Processing dynamic file: {filename} and static file: {filename} ---")

        # 训练集成模型
        try:
            integrated_model, train_dataset = train_integrated_model(
                dynamic_file_path=dynamic_file_path,
                static_file_path=static_file_path,
                results_dir=results_dir,
                dynamic_feature_cols=dynamic_feature_cols,
                static_feature_cols=static_feature_cols,
                target_col=target_col,
                window_size=10,      # 保持与GLM-Attn一致
                epochs=300,          # 增加训练轮数
                batch_size=64,        # 调整 batch size
                learning_rate=0.0003,  # 调整学习率
                dynamic_hidden_size=96,  # 调整动态模型隐藏层大小
                dynamic_num_layers=3,  # 增加LSTM层数
                static_cnn_hidden_size=64,  # 调整CNN中间层大小
                static_cnn_kernel_size=5   # 尝试不同的kernel size
            )
            print(f"Successfully trained integrated model for {filename}.")
        except Exception as e:
            print(f"Error training integrated model for {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue