import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            x_data = torch.load(kwargs["x_data"]).to(my_device).to(torch.float32)
            y_data = torch.load(kwargs["y_data"]).to(my_device).to(torch.float32)
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

    # 评估模型
    evaluate_mlp(ensemble, val_dataloader, my_device)


def evaluate_mlp(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred_y, _ = model(x)
            all_preds.extend(pred_y.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f"均方误差 (MSE): {mse}")
    print(f"平均绝对误差 (MAE): {mae}")
    print(f"决定系数 (R^2): {r2}")


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
    