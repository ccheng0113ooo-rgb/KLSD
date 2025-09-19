import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jak.common import seed_all


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=128, depth=1, dropout=0.2, device="cpu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.depth = depth
        self.my_device = device
        self.dropout = dropout

        self.model = nn.Sequential(nn.Linear(self.in_dim, self.h_dim), nn.ReLU())
        for ii in range(depth):
            self.model.add_module(f"hid_{ii}",
                                  nn.Sequential(nn.Dropout(p=self.dropout),
                                                nn.Linear(self.h_dim, self.h_dim), nn.ReLU()))
        self.model.add_module("output_layer",
                              nn.Sequential(nn.Linear(self.h_dim, self.out_dim)))

    def forward(self, x):
        x = torch.tensor(np.array(x), device=self.my_device) if type(x) is not torch.Tensor else x
        return self.model(x)


class MLPCohort(nn.Module):
    def __init__(self, in_dim, out_dim, cohort_size=3, h_dim=128, depth=3, dropout=0.2,
                 lr=3e-4, tag="no_tag", device="cpu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.depth = depth
        self.cohort_size = cohort_size
        self.lr = lr
        self.my_device = device
        self.dropout = dropout
        self.tag = tag

        self.models = []
        for ii in range(self.cohort_size):
            self.add_model()
        self.add_optimizer()
        self.to(self.my_device)

    def add_model(self):
        new_model = SimpleMLP(self.in_dim, self.out_dim, self.h_dim, self.depth,
                              self.dropout, self.my_device).to(self.my_device)
        self.models.append(new_model)
        for ii, param in enumerate(self.models[-1].parameters()):
            self.register_parameter(f"model{len(self.models)}_param{ii}", param)
        if len(self.models) > self.cohort_size:
            self.cohort_size += 1

    def add_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        """
        during training returns the prediction from a single (randomly selected) model, 
        during inference returns the mean and std. dev. of the model ensemble
        """
        x = torch.tensor(np.array(x), device=self.my_device) if type(x) is not torch.Tensor else x
        if self.training:
            model_index = torch.randint(self.cohort_size, (1,)).item()
            return self.models[model_index](x)
        else:
            with torch.no_grad():
                y = torch.Tensor().to(x.device)
                for ii in range(self.cohort_size):
                    y = torch.cat([y, self.models[ii](x).unsqueeze(0)])
                y_mean = torch.mean(y, axis=0)
                y_std_dev = torch.std(y, axis=0)
            return y_mean, y_std_dev

    def forward_ensemble(self, x):
        with torch.no_grad():
            y = torch.Tensor()
            for ii in range(self.cohort_size):
                y = torch.cat([y, self.models[ii](x).unsqueeze(0)])
        return y

    def sparse_loss(self, pred_y, target_y, dont_care=None):
        if dont_care is None:
            dont_care = 1.0 * (target_y == 0)
            do_care = 1.0 - dont_care
        loss = torch.sqrt(F.mse_loss(pred_y * do_care, target_y * do_care))
        return loss

    def training_step(self, batch_x, batch_y):
        self.train()
        pred_y = self.forward(batch_x)
        loss = self.sparse_loss(pred_y, batch_y)
        return loss

    def validation(self, validation_dataloader):
        self.eval()
        sum_loss = 0.0
        sum_std_dev = 0.0
        with torch.no_grad():
            for ii, batch in enumerate(validation_dataloader):
                pred_y, std_dev = self.forward(batch[0])
                sum_loss += self.sparse_loss(pred_y, batch[1])
                sum_std_dev += torch.mean(std_dev)
            mean_std_dev = sum_std_dev / (ii + 1)
            mean_loss = sum_loss / (ii + 1)
        return mean_loss, mean_std_dev

    def fit(self, dataloader, max_epochs, validation_dataloader=None, verbose=True):
        display_every = 1
        save_every = max(1, max_epochs // 8)
        smooth_loss = None
        alpha = 1.0 - 1e-3
        t0 = time.time()
        print("epoch, wall_time, smooth_loss, train_loss, train_std_dev, val_loss, val_std_dev")
        for epoch in range(max_epochs):
            t1 = time.time()
            sum_loss = 0.0
            for batch_number, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                loss = self.training_step(batch[0], batch[1])
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu()
                if smooth_loss is None:
                    smooth_loss = loss.detach().cpu()
                else:
                    smooth_loss = alpha * smooth_loss + (1. - alpha) * loss.detach().cpu()
            mean_loss = sum_loss / (batch_number + 1)
            t2 = time.time()
            save_filepath = os.path.join(os.getcwd(), f"parameters/{self.tag}/tag_{self.tag}_epoch{epoch}.pt")
            try:
                os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
                torch.save(self.state_dict(), save_filepath)
            except Exception as e:
                print(f"保存模型参数时出错: {e}")
            if epoch % display_every == 0 and verbose:
                elapsed_total = t2 - t0
                msg = f"{epoch}, {elapsed_total:.5e}, {smooth_loss:.5e}, {mean_loss:.5e}, "
                if validation_dataloader is not None:
                    validation_loss, val_std_dev = self.validation(validation_dataloader)
                    train_loss, train_std_dev = self.validation(dataloader)
                    msg += f"{train_std_dev:.5e}, {validation_loss:.5e}, {val_std_dev:.5e}"
                else:
                    validation_loss, val_std_dev = None, None
                    msg += f" , , , "
                print(msg)


def mlp_train(**kwargs):
    batch_size = kwargs["batch_size"]
    cohort_size = kwargs["cohort_size"]
    depth = kwargs["depth"]
    lr = kwargs["lr"]
    max_epochs = kwargs["max_epochs"]
    my_seed = kwargs["seed"]
    my_device = kwargs["device"]
    tag = f"{kwargs['tag']}_seed{my_seed}"

    # 检查 GPU 可用性
    if my_device == "cuda" and not torch.cuda.is_available():
        my_device = "cpu"
        print("CUDA 不可用，使用 CPU 进行训练。")
    else:
        print(f"使用 {my_device} 进行训练。")

    try:
        if kwargs["x_data"].endswith("pt"):
            x_data = torch.load(kwargs["x_data"]).to(my_device)
            y_data = torch.load(kwargs["y_data"]).to(my_device)
        else:
            x_data = np.load(kwargs["x_data"])
            y_data = np.load(kwargs["y_data"])
            x_data = torch.tensor(x_data, dtype=torch.float32).to(my_device)
            y_data = torch.tensor(y_data, dtype=torch.float32).to(my_device)
    except FileNotFoundError:
        print(f"数据文件未找到，请检查路径: {kwargs['x_data']} 或 {kwargs['y_data']}")
        return

    in_dim = x_data.shape[-1]
    out_dim = y_data.shape[-1]
    validation_size = int(x_data.shape[0] * 0.1)

    # 保存模型超参数
    json_filepath = os.path.join(os.getcwd(), f"parameters/{tag}/exp_tag_{tag}.json")
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
    try:
        with open(json_filepath, "w") as f:
            json.dump(kwargs, f)
    except Exception as e:
        print(f"保存超参数时出错: {e}")

    # 随机打乱数据并划分训练集和验证集
    seed_all(my_seed)
    indices = torch.randperm(x_data.shape[0])
    x_data = x_data[indices]
    y_data = y_data[indices]
    train_x = x_data[:-validation_size]
    val_x = x_data[-validation_size:]
    train_y = y_data[:-validation_size]
    val_y = y_data[-validation_size:]

    # 设置数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    ensemble = MLPCohort(in_dim, out_dim,
                         cohort_size=cohort_size,
                         depth=depth,
                         tag=tag,
                         lr=lr, device=my_device)
    ensemble.fit(train_dataloader, max_epochs, validation_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("-c", "--cohort_size", type=int, default=10,
                        help="number of mlps in ensemble/cohort")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="cpu or cuda")
    parser.add_argument("-p", "--depth", type=int, default=2,
                        help="depth of mlps in ensemble")
    parser.add_argument("-l", "--lr", type=float, default=3e-5,
                        help="learning rate")
    parser.add_argument("-m", "--max_epochs", type=int, default=100,
                        help="number of epochs to train")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="seed for pseudorandom number generators")
    parser.add_argument("-t", "--tag", type=str, default="default_tag",
                        help="string tag used to identify training run")
    parser.add_argument("-x", "--x_data", type=str, default=os.path.join(os.getcwd(), r"data\jakinase_x.pt"),
                        help="relative filepath for training input data")
    parser.add_argument("-y", "--y_data", type=str, default=os.path.join(os.getcwd(), r"data\jakinase_y.pt"),
                        help="relative filepath for training target data")
    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())
    mlp_train(**kwargs)
    

# import argparse
# import os
# import json
# import time
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from jak.common import seed_all
# import matplotlib.pyplot as plt

# # 定义 SimpleMLP 类
# class SimpleMLP(nn.Module):
#     def __init__(self, in_dim, out_dim, h_dim=512, depth=5, dropout=0.5, device="cpu"):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.h_dim = h_dim
#         self.depth = depth
#         self.my_device = device
#         self.dropout = dropout

#         self.model = nn.Sequential(nn.Linear(self.in_dim, self.h_dim), nn.ReLU())
#         for ii in range(depth):
#             self.model.add_module(f"hid_{ii}",
#                                   nn.Sequential(nn.Dropout(p=self.dropout),
#                                                 nn.Linear(self.h_dim, self.h_dim), nn.ReLU()))
#         self.model.add_module("output_layer",
#                               nn.Sequential(nn.Linear(self.h_dim, self.out_dim)))
#         self.to(self.my_device)  # 将模型移动到指定设备

#     def forward(self, x):
#         x = torch.tensor(np.array(x), device=self.my_device) if type(x) is not torch.Tensor else x.to(self.my_device)
#         output = self.model(x)
#         return output.unsqueeze(-1)  # 将输出形状从 [] 调整为 [1]

# # 定义 MLPCohort 类
# class MLPCohort(nn.Module):
#     def __init__(self, in_dim, out_dim, cohort_size=10, h_dim=512, depth=5, dropout=0.5,
#                  lr=1e-4, tag="no_tag", device="cpu"):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.h_dim = h_dim
#         self.depth = depth
#         self.cohort_size = cohort_size
#         self.lr = lr
#         self.my_device = device
#         self.dropout = dropout
#         self.tag = tag

#         self.models = []
#         for ii in range(self.cohort_size):
#             self.add_model()

#         self.add_optimizer()

#         self.to(self.my_device)

#     def add_model(self):
#         new_model = SimpleMLP(self.in_dim, self.out_dim, self.h_dim, self.depth,
#                               self.dropout, self.my_device)
#         self.models.append(new_model)

#         for ii, param in enumerate(self.models[-1].parameters()):
#             self.register_parameter(f"model{len(self.models)}_param{ii}", param)

#         if len(self.models) > self.cohort_size:
#             self.cohort_size += 1

#     def add_optimizer(self):
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

#     def forward(self, x):
#         x = x.to(self.my_device)  # 将输入数据移动到指定设备
#         if self.training:
#             model_index = torch.randint(self.cohort_size, (1,)).item()
#             return self.models[model_index](x)
#         else:
#             with torch.no_grad():
#                 y = torch.Tensor().to(self.my_device)
#                 for ii in range(self.cohort_size):
#                     y = torch.cat([y, self.models[ii](x).unsqueeze(0)])
#                 y_mean = torch.mean(y, axis=0)
#                 y_std_dev = torch.std(y, axis=0)
#             return y_mean, y_std_dev

#     # def sparse_loss(self, pred_y, target_y, dont_care=None):
#     #     pred_y = pred_y.to(self.my_device)
#     #     target_y = target_y.to(self.my_device)
#     #     if pred_y.dim() > target_y.dim():
#     #         pred_y = pred_y.squeeze()

#     #     if dont_care is None:
#     #         dont_care = 1.0 * (target_y == 0)
#     #         do_care = 1.0 - dont_care

#     #     loss = torch.sqrt(F.mse_loss(pred_y * do_care, target_y * do_care))
#     #     return loss

#     def sparse_loss(self, pred_y, target_y, dont_care=None):
#         pred_y = pred_y.to(self.my_device)
#         target_y = target_y.to(self.my_device)
#         if pred_y.dim() > target_y.dim():
#             pred_y = pred_y.squeeze()

#         if dont_care is None:
#             dont_care = 1.0 * (target_y == 0)
#             do_care = 1.0 - dont_care

#         # 使用均方误差
#         loss = F.mse_loss(pred_y * do_care, target_y * do_care)
#         return loss

#     def training_step(self, batch_x, batch_y):
#         self.train()
#         pred_y = self.forward(batch_x)
#         loss = self.sparse_loss(pred_y, batch_y)
#         return loss

#     def validation(self, validation_dataloader):
#         self.eval()
#         sum_loss = 0.0
#         sum_std_dev = 0.0

#         with torch.no_grad():
#             for ii, batch in enumerate(validation_dataloader):
#                 batch_x, batch_y = batch
#                 batch_x = batch_x.to(self.my_device)
#                 batch_y = batch_y.to(self.my_device)
#                 pred_y, std_dev = self.forward(batch_x)
#                 sum_loss += self.sparse_loss(pred_y, batch_y)
#                 sum_std_dev += torch.mean(std_dev)

#             mean_std_dev = sum_std_dev / (ii + 1)
#             mean_loss = sum_loss / (ii + 1)

#         return mean_loss, mean_std_dev

#     def fit(self, dataloader, max_epochs, validation_dataloader=None, verbose=True):
#         display_every = 1
#         save_every = max(1, max_epochs // 8)

#         smooth_loss = None
#         alpha = 1.0 - 1e-3
#         t0 = time.time()

#         # 学习率调度器
#         scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

#         # 早停法
#         best_loss = float('inf')
#         patience = 200
#         no_improvement = 0

#         print("epoch, wall_time, smooth_loss, train_loss, train_std_dev, val_loss, val_std_dev")

#         for epoch in range(max_epochs):
#             t1 = time.time()
#             sum_loss = 0.0
#             for batch_number, batch in enumerate(dataloader):
#                 batch_x, batch_y = batch
#                 batch_x = batch_x.to(self.my_device)
#                 batch_y = batch_y.to(self.my_device)
#                 self.optimizer.zero_grad()
#                 loss = self.training_step(batch_x, batch_y)
#                 loss.backward()
#                 self.optimizer.step()
#                 sum_loss += loss.detach().cpu()

#                 if smooth_loss is None:
#                     smooth_loss = loss.detach().cpu()
#                 else:
#                     smooth_loss = alpha * smooth_loss + (1. - alpha) * loss.detach().cpu()

#             # 保存模型参数
#             if epoch == max_epochs - 1:  # 在最后一个 epoch 保存模型
#                 save_path = f"parameters/{self.tag}/mlp_model.pt"
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                 torch.save(self.state_dict(), save_path)
#                 print(f"模型已保存到 {save_path}")

#             mean_loss = sum_loss / (batch_number + 1)
#             t2 = time.time()

#             save_filepath = f"parameters/{self.tag}/tag_{self.tag}_epoch{epoch}.pt"
#             if epoch % save_every == 0 or epoch == max_epochs - 1:
#                 os.makedirs(os.path.split(save_filepath)[0], exist_ok=True)
#                 torch.save(self.state_dict(), save_filepath)

#             if epoch % display_every == 0 and verbose:
#                 elapsed_total = t2 - t0
#                 msg = f"{epoch}, {elapsed_total:.5e}, {smooth_loss:.5e}, {mean_loss:.5e}, "
#                 if validation_dataloader is not None:
#                     validation_loss, val_std_dev = self.validation(validation_dataloader)
#                     train_loss, train_std_dev = self.validation(dataloader)
#                     msg += f"{train_std_dev:.5e}, {validation_loss:.5e}, {val_std_dev:.5e}"
#                     # 更新学习率调度器
#                     scheduler.step(validation_loss)
#                     # 早停法
#                     if validation_loss < best_loss:
#                         best_loss = validation_loss
#                         no_improvement = 0
#                     else:
#                         no_improvement += 1
#                         if no_improvement >= patience:
#                             print("早停法：验证损失不再下降，停止训练。")
#                             break
#                 else:
#                     validation_loss, val_std_dev = None, None
#                     msg += f" , , , "
#                 print(msg)

# def evaluate(model, test_dataloader, device):
#     model.eval()
#     mse_loss = nn.MSELoss()
#     total_mse = 0.0
#     total_mae = 0.0
#     total_r2 = 0.0
#     num_batches = 0
#     num_valid_batches = 0  # 统计有效批次数

#     with torch.no_grad():
#         for batch_x, batch_y in test_dataloader:
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             pred_y, _ = model(batch_x)

#             # 确保 pred_y 和 batch_y 的形状一致
#             if pred_y.shape != batch_y.shape:
#                 pred_y = pred_y.view_as(batch_y)

#             # 检查 batch_y 和 pred_y 是否包含 NaN 或 Inf
#             print("batch_y 是否包含 NaN 或 Inf:", torch.isnan(batch_y).any(), torch.isinf(batch_y).any())
#             print("pred_y 是否包含 NaN 或 Inf:", torch.isnan(pred_y).any(), torch.isinf(pred_y).any())

#             # 计算均方误差
#             mse = mse_loss(pred_y, batch_y)
#             total_mse += mse.item()

#             # 计算平均绝对误差
#             mae = F.l1_loss(pred_y, batch_y)
#             total_mae += mae.item()

#             # 计算 R²（手动计算）
#             if len(batch_y) > 1 and torch.var(batch_y) > 0:
#                 ss_total = torch.sum((batch_y - torch.mean(batch_y)) ** 2)
#                 ss_residual = torch.sum((batch_y - pred_y) ** 2)
#                 print("ss_residual:", ss_residual.item())
#                 print("ss_total:", ss_total.item())
#                 if ss_total > 0:
#                     r2 = 1 - (ss_residual / ss_total)
#                     if not torch.isnan(r2) and not torch.isinf(r2):  # 确保 R² 有效
#                         total_r2 += r2.item()
#                         num_valid_batches += 1  # 统计有效批次数
#                     print("R²:", r2.item() if not torch.isnan(r2) else "NaN")
#                 else:
#                     r2 = float('nan')
#             else:
#                 r2 = float('nan')

#             num_batches += 1

#     # 计算均方根误差
#     rmse = np.sqrt(total_mse / num_batches)
#     mse = total_mse / num_batches
#     mae = total_mae / num_batches

#     # 计算 R²
#     if num_valid_batches > 0:
#         r2 = total_r2 / num_valid_batches
#     else:
#         r2 = float('nan')  # 如果样本数不足或目标值方差为零，返回 NaN

#     return mse, rmse, mae, r2
# # 定义训练函数
# def mlp_train(**kwargs):
#     batch_size = kwargs["batch_size"]
#     cohort_size = kwargs["cohort_size"]
#     depth = kwargs["depth"]
#     lr = kwargs["lr"]
#     max_epochs = kwargs["max_epochs"]
#     my_seed = kwargs["seed"]
#     my_device = kwargs["device"]

#     # 检查 GPU 可用性
#     if my_device == "cuda" and not torch.cuda.is_available():
#         my_device = "cpu"
#         print("CUDA 不可用，使用 CPU 进行训练。")
#     else:
#         print(f"使用 {my_device} 进行训练。")

#     tag = f"{kwargs['tag']}_seed{my_seed}"

#     # 加载 Transformer 模型提取的特征表示
#     x_data = np.load("data/encoded_features.npy", allow_pickle=True)  # 特征表示文件
#     y_data = np.load("data/target_values.npy", allow_pickle=True)  # 目标值文件

#     # 检查 y_data 的内容和数据类型
#     print("y_data 数据类型:", y_data.dtype)
#     print("y_data 示例:", y_data[:5])  # 打印前 5 个样本

#     # 转换 y_data 的数据类型
#     if y_data.dtype == np.object_:
#         # 如果 y_data 是类别标签，使用 LabelEncoder 转换为整数
#         from sklearn.preprocessing import LabelEncoder
#         label_encoder = LabelEncoder()
#         y_data = label_encoder.fit_transform(y_data).astype(np.float32)
#     else:
#         # 如果 y_data 是数值类型，直接转换为 float32
#         y_data = y_data.astype(np.float32)

#     # 调整目标值形状
#     y_data = y_data.reshape(-1, 1)  # 将目标值形状从 [N] 调整为 [N, 1]

#     # 检查目标值的方差
#     y_var = np.var(y_data)
#     print(f"目标值的方差: {y_var}")
#     if y_var == 0:
#         print("警告：目标值的方差为零，R² 无法计算！")

#     # 检查目标值的分布
#     print("目标值的最小值:", np.min(y_data))
#     print("目标值的最大值:", np.max(y_data))
#     print("目标值的均值:", np.mean(y_data))
#     print("目标值的标准差:", np.std(y_data))

#     # 标准化特征值和目标值
#     scaler_x = StandardScaler()
#     x_data = scaler_x.fit_transform(x_data)

#     scaler_y = StandardScaler()
#     y_data = scaler_y.fit_transform(y_data).flatten()

#     # 检查标准化后的目标值
#     print("标准化后的目标值示例:", y_data[:5])

#     # 计算输入维度
#     in_dim = x_data.shape[1]  # 特征表示的维度
#     out_dim = 1  # 目标值是单值

#     # 划分数据集：训练集 80%，验证集 10%，测试集 10%
#     validation_size = int(x_data.shape[0] * 0.1)
#     test_size = max(2, int(x_data.shape[0] * 0.1))  # 确保测试集至少有 2 个样本

#     # 检查测试集的样本数量
#     print("测试集的样本数量:", test_size)
#     if test_size < 2:
#         print("警告：测试集样本数量不足，R² 无法计算！")

#     # 保存模型超参数
#     json_filepath = f"parameters/{tag}/exp_tag_{tag}.json"
#     os.makedirs(os.path.split(json_filepath)[0], exist_ok=True)
#     with open(json_filepath, "w") as f:
#         json.dump(kwargs, f)

#     # 随机打乱数据并划分训练集、验证集和测试集
#     seed_all(my_seed)
#     indices = torch.randperm(x_data.shape[0])
#     x_data = torch.tensor(x_data[indices], dtype=torch.float32).to(my_device)
#     y_data = torch.tensor(y_data[indices], dtype=torch.float32).to(my_device)

#     train_x = x_data[:-(validation_size + test_size)]
#     val_x = x_data[-(validation_size + test_size):-test_size]
#     test_x = x_data[-test_size:]

#     train_y = y_data[:-(validation_size + test_size)]
#     val_y = y_data[-(validation_size + test_size):-test_size]
#     test_y = y_data[-test_size:]

#     # 设置数据加载器
#     train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
#     val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
#     test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

#     train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
#     val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
#     test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#     # 初始化 MLP 模型
#     ensemble = MLPCohort(in_dim, out_dim,
#                          cohort_size=cohort_size,
#                          depth=depth,
#                          tag=tag,
#                          lr=lr, device=my_device)

#     # 训练模型
#     ensemble.fit(train_dataloader, max_epochs, validation_dataloader=val_dataloader)

#     # 评估模型
#     mse, rmse, mae, r2 = evaluate(ensemble, test_dataloader, my_device)
#     print(f"均方误差 (MSE): {mse:.5e}")
#     print(f"均方根误差 (RMSE): {rmse:.5e}")
#     print(f"平均绝对误差 (MAE): {mae:.5e}")
#     print(f"决定系数 (R²): {r2:.5f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-b", "--batch_size", type=int, default=64,
#                         help="batch size")
#     parser.add_argument("-c", "--cohort_size", type=int, default=10,
#                         help="number of mlps in ensemble/cohort")
#     parser.add_argument("-d", "--device", type=str, default="cuda",
#                         help="cpu or cuda")
#     parser.add_argument("-p", "--depth", type=int, default=5,
#                         help="depth of mlps in ensemble")
#     parser.add_argument("-l", "--lr", type=float, default=1e-3,
#                         help="learning rate")
#     parser.add_argument("-m", "--max_epochs", type=int, default=500,
#                         help="number of epochs to train")
#     parser.add_argument("-s", "--seed", type=int, default=42,
#                         help="seed for pseudorandom number generators")
#     parser.add_argument("-t", "--tag", type=str, default="default_tag",
#                         help="string tag used to identify training run")
#     parser.add_argument("-x", "--x_data", type=str, default="data/encoded_features.npy",
#                         help="relative filepath for training input data")
#     parser.add_argument("-y", "--y_data", type=str, default="data/target_values.npy",
#                         help="relative filepath for training target data")

#     args = parser.parse_args()
#     kwargs = dict(args._get_kwargs())

#     mlp_train(**kwargs)




