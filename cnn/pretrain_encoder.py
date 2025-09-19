# import pandas as pd
# import torch
# import os
# from tqdm import tqdm
# # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from encoder import encoder
# import pandas as pd
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from rdkit.Chem import rdMolDescriptors
# from sklearn.model_selection import KFold
# from sklearn.metrics import classification_report
# from collections import Counter
# import matplotlib.pyplot as plt
# import numpy as np
# import rdkit
# from rdkit import Chem
# from const import *


# def smiles2adj(sml):
#     """Argument for the RD2NX function should be a valid SMILES sequence
#     returns: the graph
#     """
#     # atom2num={6:0, 7:1, 8:2, 9:3, 11:4, 14:5, 15:6, 16:7, 17:8, 19:9, 35:10, 53:11, 5:12, 1:13}
#     m = rdkit.Chem.MolFromSmiles(sml)
#     # m = rdkit.Chem.AddHs(m)
#     order_string = {
#         rdkit.Chem.rdchem.BondType.SINGLE: 1,
#         rdkit.Chem.rdchem.BondType.DOUBLE: 2,
#         rdkit.Chem.rdchem.BondType.TRIPLE: 3,
#         rdkit.Chem.rdchem.BondType.AROMATIC: 4,
#     }
#     try:
#         N = len(list(m.GetAtoms()))
#     except:
#         print(sml)

#     return rdMolDescriptors.GetMACCSKeysFingerprint(m)

#     # adj = np.zeros((adj_max, adj_max))
#     # for j in m.GetBonds():
#     #     u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
#     #     v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
#     #     order = j.GetBondType()
#     #     if order in order_string:
#     #         order = order_string[order]
#     #     else:
#     #         raise Warning("Ignoring bond order" + order)
#     #     if u < 80 and v < 80:
#     #         # u=m.GetAtomWithIdx(u).GetAtomicNum()
#     #         # v=m.GetAtomWithIdx(v).GetAtomicNum()
#     #         # print(u,v)
#     #         adj[u, v] = 1#order
#     #         # if u!=v:
#     #         adj[v, u] = 1#order
#     # # nodes[:,1] =adj.sum(axis=0)
#     # # adj += np.eye(N)
#     # return adj


# class pretrain_dataset(Dataset):
#     def __init__(self, dataframe, max_len=max_len):
#         super(pretrain_dataset, self).__init__()

#         self.len = len(dataframe)
#         self.dataframe = dataframe
#         self.max_len = max_len

#     def __getitem__(self, idx):
#         X = torch.zeros(self.max_len)
#         sml = self.dataframe.canonical_smiles[idx]
#         # print(X)
#         for idx, atom in enumerate(list(sml)[:self.max_len]):
#             X[idx] = vocabulary[atom]
#         chem_id = self.dataframe.chembl_id[idx]
#         s = self.dataframe.fps[idx]
#         s = list(s)
#         adj = torch.tensor([int(b) for b in s])
#         # adj = torch.tensor(smiles2adj(sml))
#         # print(X,adj,chem_id)
#         return X, adj, chem_id

#     def __len__(self):
#         return self.len


# if __name__ == '__main__':
#     df = pd.read_csv('processed_data/chembl_30_chemreps.txt', sep='\t')
#     params = {'batch_size': 1024, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
#     data = pretrain_dataset(df)
#     loader = DataLoader(data, **params)

#     # e = encoder(max_len, len(vocabulary)).cuda()
#     e = encoder(max_len, len(vocabulary)).to('cpu')

#     load_epoch = -1
#     if load_epoch != -1:
#         # e.load_state_dict(
#         #     torch.load('checkpoints/CNN_encoder_pretrain' + str(load_epoch) + ".pt", map_location=device))
#         e.load_state_dict(
#             torch.load('checkpoints/CNN_encoder_pretrain' + str(load_epoch) + ".pt", map_location=torch.device('cpu')))


#     optimizer = optim.SGD(params=e.parameters(), lr=0.9, weight_decay=1e-2)
#     loss_function = nn.BCEWithLogitsLoss()

#     epoch = 20
#     loss_list = []
#     for i in range(load_epoch + 1, epoch):
#         total_loss = 0
#         for idx, (X, adj, chem_id) in tqdm(enumerate(loader), total=len(loader)):
#             optimizer.zero_grad()
#             # X = X.cuda().long()
#             # adj = adj.float().cuda()
#             X = X.to('cpu').long()
#             adj = adj.float().to('cpu')

#             output = e(X)
#             # print(X.shape,output.shape)
#             loss = loss_function(output, adj)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print("epoch ", i, "loss", total_loss)
#         loss_list.append(total_loss / len(loader))
#         torch.save(e.state_dict(), 'checkpoints/CNN_encoder_pretrain' + str(i) + ".pt")

# import matplotlib.pyplot as plt

# # 绘制损失曲线
# plt.plot(range(len(loss_list)), loss_list, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.show()



from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# 数据预处理函数
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    """将 SMILES 字符串转换为分子指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return fingerprint

# 数据集类
class PretrainDataset(Dataset):
    def __init__(self, dataframe, max_len=128):
        super(PretrainDataset, self).__init__()
        self.dataframe = dataframe
        self.max_len = max_len

    def __getitem__(self, idx):
        smiles = self.dataframe.canonical_smiles[idx]
        chem_id = self.dataframe.chembl_id[idx]
        return smiles, chem_id

    def __len__(self):
        return len(self.dataframe)

# CNN 模型
class CNNModel(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(32 * (input_size - 2), output_size)  # 假设分子指纹长度为 1024

    def forward(self, smiles_list):
        fingerprints = []
        for smiles in smiles_list:
            fingerprint = smiles_to_fingerprint(smiles)
            if fingerprint is not None:
                fingerprints.append(torch.tensor(fingerprint, dtype=torch.float32))
        if len(fingerprints) == 0:
            raise ValueError("所有 SMILES 字符串转换指纹时失败。")
        fingerprints = torch.stack(fingerprints)
        # 将指纹数据移动到 GPU 上
        fingerprints = fingerprints.to(device)
        x = fingerprints.unsqueeze(1)  # 增加一个维度以适应卷积层
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# 加载数据集
df = pd.read_csv(r'D:\CC\PycharmProjects\GoGT\JAK_ML\pretrained_data\chembl_30_chemreps.txt', sep='\t')
params = {'batch_size': 32, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
dataset = PretrainDataset(df)
loader = DataLoader(dataset, **params)

# 初始化模型
model = CNNModel(input_size=1024, output_size=1024)
if cuda_available:
    model = model.to(device)

# 加载预训练模型（如果存在）
load_epoch = -1
if load_epoch != -1:
    model.load_state_dict(
        torch.load('checkpoints/cnn_pretrain_' + str(load_epoch) + ".pt", map_location=device)
    )

# 优化器和学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 损失函数
loss_function = nn.BCEWithLogitsLoss()
# 将损失函数移动到 GPU 上
loss_function = loss_function.to(device)

# 训练
epoch = 30
loss_list = []
for i in range(load_epoch + 1, epoch):
    total_loss = 0
    for idx, (smiles_list, chem_id) in tqdm(enumerate(loader), total=len(loader)):
        optimizer.zero_grad()
        output = model(smiles_list)

        # 这里需要重新计算指纹作为目标值
        target_fingerprints = []
        for smiles in smiles_list:
            fingerprint = smiles_to_fingerprint(smiles)
            if fingerprint is not None:
                target_fingerprints.append(torch.tensor(fingerprint, dtype=torch.float32))
        if len(target_fingerprints) == 0:
            continue
        target_fingerprints = torch.stack(target_fingerprints)
        # 将目标指纹数据移动到 GPU 上
        target_fingerprints = target_fingerprints.to(device)

        loss = loss_function(output, target_fingerprints)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_list.append(avg_loss)
    print(f"Epoch {i}, Loss: {avg_loss}")

    # 更新学习率
    scheduler.step(avg_loss)

    # 创建保存模型的目录（如果不存在）
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 保存模型
    torch.save(model.state_dict(), 'checkpoints/cnn_pretrain_' + str(i) + ".pt")

# 绘制损失曲线
plt.plot(range(len(loss_list)), loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('cnn_training_loss.png')
plt.show()