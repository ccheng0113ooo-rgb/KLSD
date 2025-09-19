import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class XFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.batch_first = kwargs.get("batch_first", True)
        # 确保使用统一的词汇表
        self.vocab = kwargs.get("vocab", "#()+-./123456789=@BCFHIKNOPS[]acilnors")
        self.token_dict = make_token_dict(self.vocab)
        self.d_model = len(self.vocab)
        print(f"Model vocabulary size: {self.d_model}")

        # 增加模型复杂度：层数从4增加到6，注意力头从1增加到4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4,
                                                   dim_feedforward=512, batch_first=self.batch_first)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=4,
                                                   dim_feedforward=512, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.my_device = kwargs.get("device", "cpu")
        self.mask_rate = kwargs.get("mask_rate", 0.2)
        self.lr = kwargs.get("lr", 1e-4)  # 调整学习率
        self.tag = kwargs.get("tag", "default_tag")
        self.seq_length = kwargs.get("seq_length", 128)
        self.token_dim = kwargs.get("token_dim", self.d_model)

    def add_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)  # 调整优化器超参数
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def forward_x(self, x):
        print(f"Input shape: {x.shape}")
        encoded = self.encoder(x)
        x = self.decoder(x, encoded)
        return x

    def seq2seq(self, sequence: str) -> str:
        tokens = sequence_to_vectors(sequence, self.token_dict, pad_to=self.seq_length)
        one_hot = tokens_to_one_hot(tokens, pad_to=self.seq_length, pad_classes_to=self.token_dim)
        one_hot = one_hot.to(self.my_device)
        if self.batch_first:
            pass
        else:
            one_hot = one_hot.permute(1, 0, 2)
        encoded = self.encoder(one_hot)
        decoded = self.decoder(one_hot, encoded)
        decoded_sequence = one_hot_to_sequence(decoded, self.token_dict)
        return decoded_sequence

    def encode(self, sequence: str):
        tokens = sequence_to_vectors(sequence, self.token_dict, pad_to=self.seq_length)
        one_hot = tokens_to_one_hot(tokens, pad_to=self.seq_length, pad_classes_to=self.token_dim)
        one_hot = one_hot.to(self.my_device)
        if self.batch_first:
            pass
        else:
            one_hot = one_hot.permute(1, 0, 2)
        encoded = self.encoder(one_hot)
        return encoded

    def forward(self, sequence: str):
        tokens = sequence_to_vectors(sequence, self.token_dict, pad_to=self.seq_length)
        one_hot = tokens_to_one_hot(tokens, pad_to=self.seq_length, pad_classes_to=self.token_dim)
        one_hot = one_hot.to(self.my_device)
        decoded = self.forward_x(one_hot)
        output_sequence = one_hot_to_sequence(decoded, self.token_dict)
        return output_sequence

    def calc_loss(self, masked, target):
        predicted = self.forward_x(masked)
        loss = F.cross_entropy(predicted, target)
        return loss

    def training_step(self, batch):
        one_hot = batch
        vector_mask = 1.0 * (torch.rand(*one_hot.shape[:-1], 1).to(self.my_device) > self.mask_rate)
        masked_tokens = one_hot * vector_mask
        loss = self.calc_loss(masked_tokens, one_hot)
        return loss

    def validation(self, validation_dataloader):
        sum_loss = 0.0
        with torch.no_grad():
            for ii, batch in enumerate(validation_dataloader):
                if self.batch_first:
                    validation_batch = batch[0]
                else:
                    validation_batch = batch[0].permute(1, 0, 2)
                sum_loss += self.training_step(validation_batch).cpu()
            mean_loss = sum_loss / (1 + ii)
        return mean_loss

    def fit(self, dataloader, max_epochs, validation_dataloader=None, verbose=True):
        self.add_optimizer()
        save_every = max(max_epochs // 8, 1)
        display_every = save_every
        smooth_loss = None
        alpha = 1.0 - 1e-3
        t0 = time.time()
        print("epoch, wall_time, smooth_loss, train_loss, val_loss,")
        train_losses = []
        val_losses = []

        for epoch in range(max_epochs):
            t1 = time.time()
            sum_loss = 0.0
            for batch_number, batch in enumerate(dataloader):
                if self.batch_first:
                    training_batch = batch[0]
                else:
                    training_batch = batch[0].permute(1, 0, 2)
                self.optimizer.zero_grad()
                loss = self.training_step(training_batch)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu()
                if smooth_loss is None:
                    smooth_loss = loss.detach().cpu()
                else:
                    smooth_loss = alpha * smooth_loss + (1. - alpha) * loss.detach().cpu()
            mean_loss = sum_loss / (batch_number + 1)
            t2 = time.time()
            save_filepath = f"parameters/{self.tag}/tag_{self.tag}_epoch{epoch}.pt"
            if epoch % save_every == 0 or epoch == max_epochs - 1:
                os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
                torch.save(self.state_dict(), save_filepath)
            if epoch % display_every == 0 and verbose:
                elapsed_total = t2 - t0
                train_loss = self.validation(dataloader)
                # 将损失移动到 CPU 并转换为 NumPy 数组
                train_losses.append(train_loss.cpu().numpy())
                msg = f"{epoch}, {elapsed_total:.5e}, {smooth_loss:.5e}, {train_loss:.5e}, "
                if validation_dataloader is not None:
                    self.eval()
                    validation_loss = self.validation(validation_dataloader)
                    self.scheduler.step(validation_loss)
                    # 将损失移动到 CPU 并转换为 NumPy 数组
                    val_losses.append(validation_loss.cpu().numpy())
                    self.train()
                    msg += f"{validation_loss:.5e},"
                else:
                    validation_loss = None
                    msg += f" ,"
                print(msg)

    # 绘制损失曲线
        plt.plot(train_losses, label='Training Loss')
        if validation_dataloader is not None:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=128, depth=3, dropout=0.2, device="cpu"):
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)

    def forward(self, x):
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
        loss = F.mse_loss(pred_y * do_care, target_y * do_care)  # 使用MSE损失函数
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
        train_losses = []
        val_losses = []

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
                # 将损失移动到 CPU 并转换为 NumPy 数组
                    train_losses.append(train_loss.cpu().numpy())
                    val_losses.append(validation_loss.cpu().numpy())
                    msg += f"{train_std_dev:.5e}, {validation_loss:.5e}, {val_std_dev:.5e}"
                else:
                    validation_loss, val_std_dev = None, None
                    msg += f" , , , "
                print(msg)

    # 绘制损失曲线
        plt.plot(train_losses, label='Training Loss')
        if validation_dataloader is not None:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


def xformer_train(**kwargs):
    batch_size = kwargs["batch_size"]
    lr = kwargs["lr"]
    max_epochs = kwargs["max_epochs"]
    my_seed = kwargs["seed"]
    my_device = kwargs["device"]
    tag = f"xformer_{kwargs['tag']}_seed{my_seed}"
    x_data = np.load(kwargs["x_data"])
    # 检查是否为 .npz 文件
    if isinstance(x_data, np.lib.npyio.NpzFile):
        # 假设 .npz 文件中只有一个数组，取出该数组
        keys = list(x_data.keys())
        if len(keys) == 1:
            x_data = x_data[keys[0]]
        else:
            print("无法确定 .npz 文件中要使用的数组，请检查文件内容。")
            return
    validation_size = int(x_data.shape[0] * 0.1)
    json_filepath = f"D:/desktop/JAKInhibition-master/JAKInhibition-master/parameters/{tag}/exp_tag_{tag}.json"
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
    with open(json_filepath, "w") as f:
        json.dump(kwargs, f)
    seed_all(my_seed)
    x_data = torch.tensor(x_data, dtype=torch.float32).to(my_device)
    train_x = x_data[:-validation_size]
    val_x = x_data[-validation_size:]
    train_dataset = torch.utils.data.TensorDataset(train_x)
    val_dataset = torch.utils.data.TensorDataset(val_x)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    smiles_vocab = "#()+-./123456789=@BCFHIKNOPS[]acilnors"
    xformer = XFormer(lr=lr, tag=tag, device=my_device, vocab=smiles_vocab)
    xformer.to(my_device)
    xformer.fit(train_dataloader, max_epochs, validation_dataloader=val_dataloader)


def mlp_train(**kwargs):
    batch_size = kwargs["batch_size"]
    cohort_size = kwargs["cohort_size"]
    depth = kwargs["depth"]
    lr = kwargs["lr"]
    max_epochs = kwargs["max_epochs"]
    my_seed = kwargs["seed"]
    my_device = kwargs["device"]
    tag = f"{kwargs['tag']}_seed{my_seed}"
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

    # 数据标准化
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data.cpu().numpy())
    x_data = torch.tensor(x_data, dtype=torch.float32).to(my_device)

    # 特征降维
    pca = PCA(n_components=64)
    x_data = pca.fit_transform(x_data.cpu().numpy())
    x_data = torch.tensor(x_data, dtype=torch.float32).to(my_device)

    in_dim = x_data.shape[-1]
    out_dim = y_data.shape[-1]
    validation_size = int(x_data.shape[0] * 0.1)
    json_filepath = os.path.join(os.getcwd(), f"parameters/{tag}/exp_tag_{tag}.json")
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
    try:
        with open(json_filepath, "w") as f:
            json.dump(kwargs, f)
    except Exception as e:
        print(f"保存超参数时出错: {e}")
    seed_all(my_seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=my_seed)  # 初始化 KFold 对象
    fold_scores = []
    for fold, (train_index, val_index) in enumerate(kf.split(x_data)):
        train_x = x_data[train_index]
        val_x = x_data[val_index]
        train_y = y_data[train_index]
        val_y = y_data[val_index]
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
        val_loss, _ = ensemble.validation(val_dataloader)
        # 将损失移动到 CPU 并转换为 NumPy 数组
        fold_scores.append(val_loss.cpu().numpy())
        print(f"Fold {fold + 1} Validation Loss: {val_loss}")

    # 计算平均损失
    print(f"5-Fold Cross Validation Average Loss: {np.mean(fold_scores)}")


# 假设的辅助函数
def make_token_dict(vocab):
    return {token: i for i, token in enumerate(vocab)}


def sequence_to_vectors(sequence, token_dict, pad_to):
    # 简单示例实现
    tokens = [token_dict.get(token, 0) for token in sequence]
    if len(tokens) < pad_to:
        tokens.extend([0] * (pad_to - len(tokens)))
    return torch.tensor(tokens)


def one_hot_to_sequence(one_hot, token_dict):
    # 简单示例实现
    inverse_token_dict = {i: token for token, i in token_dict.items()}
    indices = torch.argmax(one_hot, dim=-1)
    sequence = ''.join([inverse_token_dict.get(index.item(), '') for index in indices])
    return sequence


def tokens_to_one_hot(tokens, pad_to, pad_classes_to):
    # 简单示例实现
    one_hot = torch.zeros((len(tokens), pad_classes_to))
    for i, token in enumerate(tokens):
        one_hot[i, token] = 1
    if len(one_hot) < pad_to:
        padding = torch.zeros((pad_to - len(one_hot), pad_classes_to))
        one_hot = torch.cat([one_hot, padding])
    return one_hot


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
    parser.add_argument("-l", "--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("-m", "--max_epochs", type=int, default=500,  # 增加训练epoch
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
    xformer_train(**kwargs)
    mlp_train(**kwargs)

    