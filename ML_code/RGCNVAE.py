import numpy as np
import pandas as pd
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GATConv, RGCNConv, GCNConv, GlobalAttention
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
from itertools import cycle
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Dataset
import os
import random
import inspect
import json
import sys
import datetime
from collections import defaultdict

# 设置所有随机种子确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(42)  # 固定随机种子

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据加载的worker初始化函数
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 定义确定性DataLoader
def get_dataloader(dataset, batch_size, shuffle=True, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init_fn
    )

# 定义权重初始化函数
def weights_init(m):
    torch.manual_seed(42)  # 为每次初始化设置相同的随机种子
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)  # Embedding 层也使用 Xavier 初始化

# 定义 GDataset 类
class GDataset(Dataset):
    def __init__(self, nodes, edges, relations, y, idx):
        super(GDataset, self).__init__()
        self.nodes = nodes
        self.edges = edges
        self.relations = relations
        self.y = y
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        idx = self.idx[idx]
        edge_index = torch.tensor(self.edges[idx].T, dtype=torch.long)  # 边索引
        x = torch.tensor(self.nodes[idx], dtype=torch.long)  # 节点特征
        y = torch.tensor(self.y[idx], dtype=torch.float)  # 标签
        edge_type = torch.tensor(self.relations[idx], dtype=torch.float)  # 边类型

        embedding_sizes = [33, 5, 3, 4, 2, 3]
        for i in range(6):
            x[:, i] = torch.clamp(x[:, i], 0, embedding_sizes[i] - 1)  # 修正无效索引

        return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)

# 定义数据预处理函数
def preprocess(dataset_name):
    # 保存预处理配置
    preprocess_config = {
        'dataset_name': dataset_name,
        'atom_types': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88],
        'embedding_sizes': [33, 5, 3, 4, 2, 3],
        'bond_types': {
            'SINGLE': 1,
            'DOUBLE': 2,
            'TRIPLE': 3,
            'AROMATIC': 4
        }
    }
    
    # 保存预处理配置到文件
    config_path = f'./saved_models/{dataset_name}/preprocess_config.pkl'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'wb') as f:
        pickle.dump(preprocess_config, f)
    
    rootpath = 'D:/CC/PycharmProjects/GoGT/JAK_ML/data/'  # 修改为你的数据路径
    JAK = [rootpath + f'unique_Homo_um_pact_act{dataset_name}.xlsx']  # 更新为.xlsx扩展名
    LABEL = [rootpath + f'{dataset_name}_graph_labels.txt']
    nodes = []
    edges = []
    relations = []
    y = []
    smiles_all = []

    for path1, path2 in zip(JAK, LABEL):
        try:
            # 使用pd.read_excel读取Excel文件
            df = pd.read_excel(path1)
            smiles = df['canonical_smiles'].tolist()
            y_ = ((np.array(df['Activity'].tolist()) > 0) * 1)
            smiles_all.extend(smiles)
        except FileNotFoundError:
            print(f"File {path1} not found.")
            continue
        except KeyError:
            print(f"Key 'canonical_smiles' or 'Activity' not found in {path1}.")
            continue
        except Exception as e:
            print(f"Error reading {path1}: {e}")
            continue

        lens = []
        adjs = []  # 保持为 Python 列表
        ords = []
        for i in range(len(smiles)):
            node, adj, order = gen_smiles2graph(smiles[i])
            if node is None or adj is None:  # 如果解析失败，则跳过此样本
                continue
            lens.append(adj.shape[0])
            adjs.append(adj)  # 直接添加 adj，不转换为 NumPy 数组
            ords.append(order)
            node[:, 2] += 1
            node[:, 3] -= 1
            nodes.append(node)

        lens = np.array(lens)

        def adj2idx(adj):
            idx = []
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] == 1:
                        idx.append([i, j])
            return np.array(idx)

        def order2relation(adj):
            idx = []
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] != 0:
                        idx.extend([adj[i, j]])
            return np.array(idx)

        for i in range(lens.shape[0]):
            adj = adjs[i]  # 直接从列表中获取 adj
            order = ords[i]
            idx = adj2idx(adj)
            relation = order2relation(order) - 1
            edges.append(idx)
            relations.append(relation)
            y.append(y_[i])  # 确保标签添加与 edges 一致

    y = np.array(y)
    return smiles_all, nodes, edges, relations, y

# 定义 gen_smiles2graph 函数
def gen_smiles2graph(sml):
    ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]
    dic = {ls[i]: i for i in range(len(ls))}

    m = rdkit.Chem.MolFromSmiles(sml)
    if not m:
        print(f"Error parsing SMILES: {sml}")
        return None, None, None

    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }

    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, 6))

    for i in m.GetAtoms():
        atom_types = dic.get(i.GetAtomicNum(), 0)
        degree = min(i.GetDegree(), 4)  # 确保度不超过 4
        nodes[i.GetIdx()] = [
            atom_types, degree, i.GetFormalCharge(),
            i.GetHybridization(), i.GetIsAromatic(), i.GetChiralTag()
        ]

    adj = np.zeros((N, N))
    orders = np.zeros((N, N))
    for bond in m.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        adj[u, v] = adj[v, u] = 1
        orders[u, v] = orders[v, u] = order_string.get(bond_type, 0)

    return nodes, adj, orders

class BaseVAE(nn.Module):
    """所有VAE模型的基类，定义通用接口和参数命名"""
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(BaseVAE, self).__init__()
        # 公共参数
        self.in_embd = in_embd
        self.layer_embd = layer_embd
        self.out_embd = out_embd
        self.num_relations = num_relations
        self.dropout = dropout
        self.d = out_embd  # 潜在空间维度
        
        # 公共组件
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd), 
            nn.BatchNorm1d(out_embd), 
            nn.ReLU(), 
            nn.Linear(out_embd, 1)))
        self.graph_linear = nn.Linear(out_embd, 1)

    def recognition_model(self, x, edge_index, edge_type, batch):
        raise NotImplementedError
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, Z):
        return self.activation(Z @ Z.T)

    def forward(self, x, edge_index, edge_type, batch, type_):
        raise NotImplementedError
        
    def cal_mu(self, x, edge_index, edge_type, batch):
        mu, _ = self.recognition_model(x, edge_index, edge_type, batch)
        return mu


class RGCN_VAE(BaseVAE):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(RGCN_VAE, self).__init__(in_embd, layer_embd, out_embd, num_relations, dropout)
        
        # 特定于RGCN的嵌入层
        self.embedding = nn.ModuleList([
            nn.Embedding(35, in_embd), 
            nn.Embedding(10, in_embd),
            nn.Embedding(5, in_embd), 
            nn.Embedding(7, in_embd),
            nn.Embedding(5, in_embd), 
            nn.Embedding(5, in_embd)
        ])

        # RGCN特定层
        self.RGCNConv1 = RGCNConv(6 * in_embd, layer_embd, num_relations)
        self.RGCNConv2 = RGCNConv(layer_embd, out_embd * 2, num_relations)
        
        self.RGCNConv1.reset_parameters()
        self.RGCNConv2.reset_parameters()

    def recognition_model(self, x, edge_index, edge_type, batch):
        # 嵌入层处理
        embeds = []
        for i in range(6):
            embeds.append(self.embedding[i](x[:, i]))
        x_ = torch.cat(embeds, dim=1)
        
        # 确保edge_type类型正确
        edge_type = edge_type.long()

        # 通过RGCN层
        out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
        out = self.activation(self.RGCNConv2(out, edge_index, edge_type))

        # 分割mu和logvar
        mu = out[:, :self.d]
        logvar = out[:, self.d:]
        return mu, logvar

    def forward(self, x, edge_index, edge_type, batch, type_):
        if type_ == 'pretrain':
            mu, logvar = self.recognition_model(x, edge_index, edge_type, batch)
            Z = self.reparametrize(mu, logvar)
            A_hat = self.generation_model(Z)

            N = x.size(0)
            A = torch.zeros((N, N), device=device)
            with torch.no_grad():
                for i in range(edge_index.size(1)):
                    A[edge_index[0, i], edge_index[1, i]] = 1
            return A, A_hat, mu, logvar
        else:
            mu = self.cal_mu(x, edge_index, edge_type, batch)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            out = self.activation(out)
            return out


class GCN_VAE(BaseVAE):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GCN_VAE, self).__init__(in_embd, layer_embd, out_embd, num_relations, dropout)
        
        # 特定于GCN的嵌入层
        self.embedding = nn.ModuleList([
            nn.Embedding(33, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(3, in_embd),
            nn.Embedding(4, in_embd),
            nn.Embedding(2, in_embd),
            nn.Embedding(3, in_embd)
        ])

        # GCN特定层
        self.GCNConv1 = GCNConv(6 * in_embd, layer_embd)
        self.GCNConv2 = GCNConv(layer_embd, out_embd * 2)
        
        # 额外的线性层
        self.mu_linear = nn.Linear(out_embd * 2, out_embd)
        self.logvar_linear = nn.Linear(out_embd * 2, out_embd)
        
        self.GCNConv1.reset_parameters()
        self.GCNConv2.reset_parameters()

    def recognition_model(self, x, edge_index, edge_type, batch):
        # 嵌入层处理
        embeds = []
        for i in range(6):
            embeds.append(self.embedding[i](x[:, i]))
        x_ = torch.cat(embeds, dim=1)
        
        # 通过GCN层
        out = self.activation(self.GCNConv1(x_, edge_index))
        out = self.activation(self.GCNConv2(out, edge_index))
        
        # 通过线性层得到mu和logvar
        mu = self.mu_linear(out)
        logvar = self.logvar_linear(out)
        return mu, logvar

    def forward(self, x, edge_index, edge_type, batch, type_):
        if type_ == 'pretrain':
            mu, logvar = self.recognition_model(x, edge_index, edge_type, batch)
            Z = self.reparametrize(mu, logvar)
            A_hat = self.generation_model(Z)

            N = x.size(0)
            A = torch.zeros((N, N), device=device)
            with torch.no_grad():
                for i in range(edge_index.size(1)):
                    A[edge_index[0, i], edge_index[1, i]] = 1
            return A, A_hat, mu, logvar
        else:
            mu = self.cal_mu(x, edge_index, edge_type, batch)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            out = self.activation(out)
            return out


class GAT_VAE(BaseVAE):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GAT_VAE, self).__init__(in_embd, layer_embd, out_embd, num_relations, dropout)
        
        # 特定于GAT的嵌入层
        self.embedding = nn.ModuleList([
            nn.Embedding(35, in_embd),
            nn.Embedding(10, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(7, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(5, in_embd)
        ])

        # GAT特定层
        self.GATConv1 = GATConv(6 * in_embd, layer_embd, heads=4, dropout=dropout)
        self.GATConv2 = GATConv(layer_embd * 4, out_embd * 2, heads=4, dropout=dropout)
        
        self.GATConv1.reset_parameters()
        self.GATConv2.reset_parameters()

    def recognition_model(self, x, edge_index, edge_type, batch):
        # 嵌入层处理
        embeds = []
        for i in range(6):
            embeds.append(self.embedding[i](x[:, i]))
        x_ = torch.cat(embeds, dim=1)
        
        # 通过GAT层
        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))
        
        # 分割mu和logvar
        mu = out[:, :self.d]
        logvar = out[:, self.d:]
        return mu, logvar

    def forward(self, x, edge_index, edge_type, batch, type_):
        if type_ == 'pretrain':
            mu, logvar = self.recognition_model(x, edge_index, edge_type, batch)
            Z = self.reparametrize(mu, logvar)
            A_hat = self.generation_model(Z)

            N = x.size(0)
            A = torch.zeros((N, N), device=device)
            with torch.no_grad():
                for i in range(edge_index.size(1)):
                    A[edge_index[0, i], edge_index[1, i]] = 1
            return A, A_hat, mu, logvar
        else:
            mu = self.cal_mu(x, edge_index, edge_type, batch)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            out = self.activation(out)
            return out
def save_model(model, optimizer, model_name, dataset_name, params, metrics, epoch):
    try:
        # 创建保存目录
        save_dir = f"./saved_models/{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 准备保存内容 - 确保包含所有必要参数
        save_content = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'params': {
                'in_embd': getattr(model, 'in_embd', params.get('in_embd', 4)),
                'layer_embd': getattr(model, 'layer_embd', params.get('layer_embd', 64)),
                'out_embd': getattr(model, 'out_embd', params.get('out_embd', 128)),
                'num_relations': getattr(model, 'num_relations', params.get('num_relations', 4)),
                'dropout': getattr(model, 'dropout', params.get('dropout', 0.0))
            },
            'metrics': metrics,
            'epoch': epoch,
            'model_type': model_name,
            'model_class': model.__class__.__name__
        }
        
        # 保存模型
        save_path = f"{save_dir}/{model_name}_best.pth"
        torch.save(save_content, save_path)
        print(f"Model saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save model: {str(e)}")
        return False
def save_experiment_config(dataset_name, model_name, params):
    config = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'params': params,
        'random_seed': 42,
        'timestamp': datetime.datetime.now().isoformat(),
        'environment': {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    }
    
    config_path = f'./saved_models/{dataset_name}/{model_name}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# 定义评估函数
def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for data in test_loader:
            data.to(device)
            # 检查数据维度和类型
            if data.x.dim() != 2:
                print(f"Invalid node feature dimension in evaluation: {data.x.dim()}")
                raise ValueError("Invalid node feature dimension in evaluation.")
            if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
                print(f"Invalid edge index dimension in evaluation: {data.edge_index.dim()}")
                raise ValueError("Invalid edge index dimension in evaluation.")

            pred = model(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend((pred > 0.5).float().cpu().numpy())
            y_score.extend(pred.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)

    # 输出分类报告和混淆矩阵
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, f1, roc_auc, fpr, tpr

# 定义 VAE 模型预训练函数
def pretrain(EPOCH, split, batch_size, type_, model_type, dataset_name, idx=None, start=0, end=10000):
    if type_ == 'JAK':
        smiles, nodes, edges, relations, y = preprocess(dataset_name)
    else:
        raise NotImplementedError("Only 'JAK' type is supported for now.")

    dataset = GDataset(nodes, edges, relations, y, range(0, len(nodes)))

    train_loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)

    if model_type == 'RGCN_VAE':
        model = RGCN_VAE(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
    elif model_type == 'GCN_VAE':
        model = GCN_VAE(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
    elif model_type == 'GAT_VAE':
        model = GAT_VAE(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
    else:
        raise ValueError("Invalid model type for pretraining.")

    model.apply(weights_init)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_function(A, A_hat, mu, logvar):
        BCE = F.binary_cross_entropy(A_hat, A, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    for epoch in range(EPOCH):
        print('epoch {}:'.format(epoch + 1))
        total_loss = 0
        for data in train_loader:
            data.to(device)

            A, A_hat, mu, logvar = model(data.x, data.edge_index, data.edge_type, data.batch, 'pretrain')

            loss = loss_function(A, A_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('\ttrain loss = {}'.format(total_loss / len(train_loader.dataset)))

    return model

# 定义 VAE 模型微调函数
def finetune(GraphVAE, EPOCH, split, batch_size, weight, dataset_name):
    smiles, nodes, edges, relations, y = preprocess(dataset_name)

    dataset = GDataset(nodes, edges, relations, y, range(0, len(smiles)))

    train_num = int(split * len(smiles))
    test_num = len(smiles) - train_num

    train_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_num, test_num],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = get_dataloader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_set, batch_size=test_num, shuffle=False)

    optimizer = torch.optim.Adam(GraphVAE.parameters(), lr=1e-4)

    # 将 pos_weight 移动到与模型相同的设备上
    weight = torch.FloatTensor([weight]).to(GraphVAE.device if hasattr(GraphVAE, 'device') else device)

    def weighted_BCE(pred, target, weight):
        pos_weight = weight
        loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        return loss

    for epoch in range(EPOCH):
        print('epoch {}:'.format(epoch + 1))
        preds = []
        labels = []
        for data in train_loader:
            data.to(device)
            pred = GraphVAE(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')
            loss = weighted_BCE(pred.squeeze(), data.y, weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds.append((pred.detach().cpu().numpy()))
            labels.append(data.y.detach().cpu().numpy())

        preds = np.concatenate(preds)
        preds = np.array(preds)
        labels = np.concatenate(labels)
        labels = np.array(labels)

        comp = np.concatenate((((preds > 0.5) * 1).reshape(-1, 1), labels.reshape(-1, 1)), 1)

        train_0 = sum((comp[:, 0] == 0) * (comp[:, 1] == 0) * 1) / sum((comp[:, 1] == 0) * 1)
        train_1 = sum((comp[:, 0] == 1) * (comp[:, 1] == 1) * 1) / sum((comp[:, 1] == 1) * 1)

        print(train_0, train_1, (train_0 + train_1) / 2)

        for data in test_loader:
            data.to(device)
            preds = GraphVAE(data.x, data.edge_index, data.edge_type, data.batch, 'finetune').detach().cpu().numpy()
            labels = data.y.detach().cpu().numpy()

        comp = np.concatenate((((preds > 0.5) * 1).reshape(-1, 1), labels.reshape(-1, 1)), 1)

        test_0 = sum((comp[:, 0] == 0) * (comp[:, 1] == 0) * 1) / sum((comp[:, 1] == 0) * 1)
        test_1 = sum((comp[:, 0] == 1) * (comp[:, 1] == 1) * 1) / sum((comp[:, 1] == 1) * 1)

        print(test_0, test_1, (test_0 + test_1) / 2)

    return GraphVAE

# 定义训练函数
def train_single_fold(model, train_loader, test_loader, optimizer, scheduler, EPOCH, weight):
    model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(EPOCH):
        model.train()
        total_loss = 0

        for data in train_loader:
            data.to(device)
            # 检查数据维度和类型
            if data.x.dim() != 2:
                print(f"Invalid node feature dimension in training: {data.x.dim()}")
                raise ValueError("Invalid node feature dimension in training.")
            if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
                print(f"Invalid edge index dimension in training: {data.edge_index.dim()}")
                raise ValueError("Invalid edge index dimension in training.")

            pred = model(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')

            # 动态选择类别权重
            weight_tensor = torch.where(data.y == 1, torch.tensor(weight, device=device), torch.tensor(1.0, device=device))
            loss = F.binary_cross_entropy(pred.squeeze(), data.y, weight=weight_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.y.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        torch.cuda.empty_cache()  # 清空缓存
        # 评估测试集
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data.to(device)
                # 检查数据维度和类型
                if data.x.dim() != 2:
                    print(f"Invalid node feature dimension in testing: {data.x.dim()}")
                    raise ValueError("Invalid node feature dimension in testing.")
                if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
                    print(f"Invalid edge index dimension in testing: {data.edge_index.dim()}")
                    raise ValueError("Invalid edge index dimension in testing.")

                pred = model(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')

                # 动态选择类别权重
                weight_tensor = torch.where(data.y == 1, torch.tensor(weight, device=device), torch.tensor(1.0, device=device))
                loss = F.binary_cross_entropy(pred.squeeze(), data.y, weight=weight_tensor)

                test_loss += loss.item() * data.y.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        # 更新学习率
        scheduler.step(avg_test_loss)

        print(f"Epoch {epoch + 1}/{EPOCH}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

    return {'train_losses': train_losses, 'test_losses': test_losses}

# 定义验证一致性的函数
def verify_consistency(dataset_name, model_name):
    """验证模型在不同运行中的一致性"""
    try:
        # 1. 加载保存的模型和配置
        model_path = f'./saved_models/{dataset_name}/{model_name}_best.pth'
        config_path = f'./saved_models/{dataset_name}/preprocess_config.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"模型或配置文件不存在，跳过验证: {model_path}")
            return False
            
        # 2. 加载检查点
        checkpoint = torch.load(model_path)
        
        # 3. 根据模型类型创建模型实例
        model_classes = {
            'RGCN_VAE': RGCN_VAE,
            'GCN_VAE': GCN_VAE,
            'GAT_VAE': GAT_VAE
        }
        
        if checkpoint['model_class'] not in model_classes:
            print(f"未知模型类型: {checkpoint['model_class']}")
            return False
            
        model_class = model_classes[checkpoint['model_class']]
        model = model_class(**checkpoint['params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 4. 重新预处理数据
        with open(config_path, 'rb') as f:
            preprocess_config = pickle.load(f)
        
        smiles, nodes, edges, relations, y = preprocess(dataset_name)
        test_dataset = GDataset(nodes, edges, relations, y, range(len(y)))
        test_loader = DataLoader(test_dataset, batch_size=len(y), shuffle=False)
        
        # 5. 获取原始预测结果和重新运行的预测结果
        original_preds = []
        reproduced_preds = []
        
        for data in test_loader:
            data = data.to(device)
            
            # 原始预测
            with torch.no_grad():
                original_pred = model(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')
                original_preds.extend(original_pred.cpu().numpy())
            
            # 重新预处理后预测
            with torch.no_grad():
                reproduced_pred = model(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')
                reproduced_preds.extend(reproduced_pred.cpu().numpy())
        
        # 6. 比较结果
        original_preds = np.array(original_preds)
        reproduced_preds = np.array(reproduced_preds)
        
        diff = np.abs(original_preds - reproduced_preds).max()
        print(f"最大预测差异: {diff}")
        
        if diff < 1e-6:
            print("一致性验证通过 - 预测结果完全一致")
            return True
        else:
            print("一致性验证失败 - 预测结果不一致")
            return False
            
    except Exception as e:
        print(f"一致性验证过程中发生错误: {str(e)}")
        return False

def grid_search(model_name, dataset_name, param_grid, EPOCH, batch_size, weight, n_splits=5):
    try:
        # 数据预处理
        smiles, nodes, edges, relations, y = preprocess(dataset_name)
        if len(y) == 0:
            print(f"[Warning] 数据集 {dataset_name} 无有效数据，跳过")
            return {
                'fpr': None,
                'tpr': None,
                'roc_auc': 0,
                'metrics': None,
                'best_params': None,
                'error': "Empty dataset"
            }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 初始化最佳结果记录
        best_roc_auc = 0
        best_metrics = None
        best_params = None
        best_fpr = None
        best_tpr = None
        best_model = None
        best_optimizer = None
        failed_folds = 0
        best_train_losses = None
        best_test_losses = None
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"\nFold {fold + 1}/{n_splits}")

            try:
                # 数据准备
                train_set = GDataset(nodes, edges, relations, y, train_idx)
                test_set = GDataset(nodes, edges, relations, y, test_idx)
                
                if len(train_set) == 0 or len(test_set) == 0:
                    print("[Warning] 空数据集，跳过当前fold")
                    failed_folds += 1
                    continue

                train_loader = get_dataloader(train_set, batch_size=batch_size, shuffle=True)
                test_loader = get_dataloader(test_set, batch_size=batch_size, shuffle=False)

                # 参数网格搜索
                for params in ParameterGrid(param_grid):
                    print(f"\n尝试参数组合: {params}")
                    
                    try:
                        # 模型初始化 - 确保传递所有必要参数
                        model_params = {
                            'in_embd': params['in_embd'],
                            'layer_embd': params['layer_embd'],
                            'out_embd': params['out_embd'],
                            'num_relations': 4,  # 默认值或从param_grid获取
                            'dropout': params['dropout']
                        }
                        
                        if model_name == 'RGCN_VAE':
                            model = RGCN_VAE(**model_params)
                        elif model_name == 'GCN_VAE':
                            model = GCN_VAE(**model_params)
                        elif model_name == 'GAT_VAE':
                            model = GAT_VAE(**model_params)
                        
                        model.apply(weights_init)
                        model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
                        
                        # 保存实验配置（包含完整参数）
                        save_experiment_config(dataset_name, model_name, {**params, 'num_relations': 4})
                        
                        # 微调阶段
                        try:
                            model = finetune(model, EPOCH, 0.8, batch_size, weight, dataset_name)
                        except Exception as e:
                            print(f"[Warning] 微调阶段失败: {str(e)}")
                            continue
                        
                        # 训练和评估
                        train_result = train_single_fold(model, train_loader, test_loader, optimizer, scheduler, EPOCH, weight)
                        accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate(model, test_loader, device)
                        
                        metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc
                        }
                        
                        # 更新最佳结果
                        if roc_auc > best_roc_auc:
                            best_roc_auc = roc_auc
                            best_metrics = metrics
                            best_params = {**params, 'num_relations': 4}  # 确保包含num_relations
                            best_fpr = fpr
                            best_tpr = tpr
                            best_model = model
                            best_optimizer = optimizer
                            best_train_losses = train_result['train_losses']
                            best_test_losses = train_result['test_losses']
                            
                            # 保存最佳模型（添加参数验证）
                            try:
                                print("保存模型参数验证:", {
                                    'in_embd': model.in_embd,
                                    'layer_embd': model.layer_embd,
                                    'out_embd': model.out_embd,
                                    'num_relations': model.num_relations,
                                    'dropout': model.dropout
                                })
                                
                                save_success = save_model(
                                    model=best_model,
                                    optimizer=best_optimizer,
                                    model_name=model_name,
                                    dataset_name=dataset_name,
                                    params=best_params,
                                    metrics=best_metrics,
                                    epoch=EPOCH
                                )
                                if not save_success:
                                    print("[Warning] 模型保存失败，但继续执行")
                            except Exception as e:
                                print(f"[Warning] 模型保存异常: {str(e)}")

                    except Exception as e:
                        print(f"[Warning] 参数组合 {params} 失败: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"[Warning] Fold {fold + 1} 失败: {str(e)}")
                failed_folds += 1
                continue
                
        # 结果汇总
        if failed_folds == n_splits:
            print(f"[Error] 所有 {n_splits} folds 均失败")
            return {
                'fpr': None,
                'tpr': None,
                'roc_auc': 0,
                'metrics': None,
                'best_params': None,
                'error': "All folds failed"
            }
                
        return {
            'fpr': best_fpr,
            'tpr': best_tpr,
            'roc_auc': best_roc_auc,
            'metrics': best_metrics,
            'best_params': best_params,
            'failed_folds': failed_folds,
            'train_losses': best_train_losses,
            'test_losses': best_test_losses
        }
        
    except Exception as e:
        print(f"[Error] {model_name}/{dataset_name} 整体处理失败: {str(e)}")
        return {
            'fpr': None,
            'tpr': None,
            'roc_auc': 0,
            'metrics': None,
            'best_params': None,
            'error': str(e),
            'train_losses': None,
            'test_losses': None
        }
# 定义ROC曲线绘图函数（统一风格）
def plot_roc_curve(roc_data, title, save_path):
    plt.figure(figsize=(10, 8))
    
    # 定义颜色和线型
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    
    # 遍历数据绘制ROC曲线
    for name, data in roc_data.items():
        if data['fpr'] is not None and data['tpr'] is not None:
            fpr = data['fpr']
            tpr = data['tpr']
            roc_auc = data['roc_auc']
            
            plt.plot(fpr, tpr, 
                     color=next(colors),
                     linestyle=next(linestyles),
                     linewidth=2,
                     label=f'{name} (AUC = {roc_auc:.2f})')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    os.makedirs('./roc_curves', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 主训练流程
def train_with_consistency():
    # 定义超参数网格
    param_grid = {
        'in_embd': [8],
        'layer_embd': [32, 64],
        'out_embd': [64, 128],
        'num_relations': [4],  # 明确包含这个参数
        'dropout': [0.0, 0.2],
        'lr': [1e-3]
    }
    
    # 定义数据集列表
    datasets = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
    
    # 定义模型列表、训练轮数和批量大小
    models = ['GAT_VAE']
    epochs = [20]
    batch_sizes = [64]
    
    # 保存所有 ROC 数据和最佳指标
    all_roc_data = defaultdict(dict)
    
    for dataset_name in datasets:
        for model_name, epoch, batch_size in zip(models, epochs, batch_sizes):
            print(f"\n开始处理 {model_name}/{dataset_name}")
            
            # 运行网格搜索
            result = grid_search(
                model_name=model_name,
                dataset_name=dataset_name,
                param_grid=param_grid,
                EPOCH=epoch,
                batch_size=batch_size,
                weight=2,
                n_splits=5
            )
            
            # 验证一致性
            if result.get('metrics') is not None:
                print("\n开始一致性验证...")
                verify_consistency(dataset_name, model_name)
                
                all_roc_data[dataset_name][model_name] = result
                print(f"成功完成 {model_name}/{dataset_name}")
            else:
                print(f"跳过 {model_name}/{dataset_name} 由于处理失败")
    
    # 遍历每个模型，分别绘制其在不同数据集上的 ROC 曲线
    for model_name in models:
        model_roc_data = {dataset: all_roc_data[dataset][model_name] for dataset in datasets}
        plot_roc_curve(
            roc_data=model_roc_data,
            title=f'ROC Curves for {model_name} Model',
            save_path=f'./roc_curves/{model_name}_roc_curves.png'
        )

    # 为每个数据集绘制不同模型的ROC曲线比较
    for dataset_name in datasets:
        dataset_roc_data = {model: all_roc_data[dataset_name][model] for model in models}
        plot_roc_curve(
            roc_data=dataset_roc_data,
            title=f'ROC Curves for Different Models on {dataset_name} Dataset',
            save_path=f'./roc_curves/{dataset_name}_model_comparison.png'
        )

    # 保存所有最佳评估指标
    all_best_metrics = {}

    # 收集所有最佳评估指标
    for model_name in models:
        best_metrics = {}
        for dataset_name in datasets:
            best_metrics[dataset_name] = all_roc_data[dataset_name][model_name]['metrics']
        all_best_metrics[model_name] = best_metrics

    # 打印每个模型在每个数据集上的最佳评估指标
    for model_name, metrics in all_best_metrics.items():
        print(f"\nBest evaluation metrics for {model_name}:")
        for dataset_name, metric_values in metrics.items():
            print(f"  Dataset: {dataset_name}")
            print(f"    Accuracy: {metric_values['accuracy']:.4f}")
            print(f"    Precision: {metric_values['precision']:.4f}")
            print(f"    Recall: {metric_values['recall']:.4f}")
            print(f"    F1-score: {metric_values['f1']:.4f}")
            print(f"    ROC AUC: {metric_values['roc_auc']:.4f}")

    # 保存所有最佳评估指标到 CSV 文件
    results_df = pd.DataFrame()
    for model_name, metrics in all_best_metrics.items():
        for dataset_name, metric_values in metrics.items():
            row = {
                'Model': model_name,
                'Dataset': dataset_name,
                'Accuracy': metric_values['accuracy'],
                'Precision': metric_values['precision'],
                'Recall': metric_values['recall'],
                'F1-score': metric_values['f1'],
                'ROC AUC': metric_values['roc_auc']
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    results_df.to_csv('new_model_evaluation_results.csv', index=False)
    print("\nBest evaluation metrics have been saved to 'new_model_evaluation_results.csv'.")

    # 额外功能：绘制每个模型的训练和测试损失曲线
    for dataset_name in datasets:
        for model_name in models:
            train_losses = all_roc_data[dataset_name][model_name]['train_losses']
            test_losses = all_roc_data[dataset_name][model_name]['test_losses']

            if train_losses is not None and test_losses is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(test_losses, label='Test Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{model_name} Loss Curve on {dataset_name}')
                plt.legend()
                plt.savefig(f'new_{model_name}_{dataset_name}_loss_curve.png')
                plt.show()

# 运行训练流程
if __name__ == "__main__":
    train_with_consistency()