###RGCN训练模型
# from pickletools import float8 
# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import Dataset
# import torch.utils.data 
# from torch_geometric.data import DataLoader
# from torch_geometric.data import Data
# from torch_geometric.nn import GATConv, RGCNConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
# from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
# import rdkit
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from itertools import compress
# import random
# from collections import defaultdict
# from sklearn.metrics import roc_curve, auc
# import pickle


# rootpath = 'Downloads/'
# device = 'cuda'

# # 定义 gen_smiles2graph
# def gen_smiles2graph(sml):
#     ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]
#     dic = {ls[i]: i for i in range(len(ls))}
    
#     m = rdkit.Chem.MolFromSmiles(sml)
#     if not m:
#         print(f"Error parsing SMILES: {sml}")  # 打印出无法解析的SMILES
#         return None, None, None  # 返回None，表示解析失败
#     # if not m:
#     #     return 'error', 'error', 'error'

#     order_string = {
#         rdkit.Chem.rdchem.BondType.SINGLE: 1,
#         rdkit.Chem.rdchem.BondType.DOUBLE: 2,
#         rdkit.Chem.rdchem.BondType.TRIPLE: 3,
#         rdkit.Chem.rdchem.BondType.AROMATIC: 4,
#     }

#     N = len(list(m.GetAtoms()))
#     nodes = np.zeros((N, 6))

#     for i in m.GetAtoms():
#         atom_types = dic.get(i.GetAtomicNum(), 0)
#         nodes[i.GetIdx()] = [
#             atom_types, i.GetDegree(), i.GetFormalCharge(),
#             i.GetHybridization(), i.GetIsAromatic(), i.GetChiralTag()
#         ]

#     adj = np.zeros((N, N))
#     orders = np.zeros((N, N))
#     for bond in m.GetBonds():
#         u = bond.GetBeginAtomIdx()
#         v = bond.GetEndAtomIdx()
#         bond_type = bond.GetBondType()
#         adj[u, v] = adj[v, u] = 1
#         orders[u, v] = orders[v, u] = order_string.get(bond_type, 0)

#     return nodes, adj, orders

# # 定义 random_scaffold_split
# def generate_scaffold(smiles, include_chirality=False):
#     scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
#     return scaffold

# def random_scaffold_split(dataset, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
#     np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

#     scaffolds = defaultdict(list)
#     for ind, smiles in enumerate(smiles_list):
#         scaffold = generate_scaffold(smiles, include_chirality=True)
#         scaffolds[scaffold].append(ind)

#     scaffold_sets = np.random.RandomState(seed).permutation(list(scaffolds.values()))

#     train_idx, valid_idx, test_idx = [], [], []
#     n_total_valid = int(np.floor(frac_valid * len(dataset)))
#     n_total_test = int(np.floor(frac_test * len(dataset)))

#     for scaffold_set in scaffold_sets:
#         if len(valid_idx) + len(scaffold_set) <= n_total_valid:
#             valid_idx.extend(scaffold_set)
#         elif len(test_idx) + len(scaffold_set) <= n_total_test:
#             test_idx.extend(scaffold_set)
#         else:
#             train_idx.extend(scaffold_set)

#     return train_idx, valid_idx, test_idx

# # 定义 weights_init
# def weights_init(m): 
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)


# def weighted_BCE(output, target, weight=None):  
#     if weight is not None:
#         # 扩展 weight 到与 target 和 output 相同的形状
#         # 假设 weight 是针对二分类问题的，每个样本会有对应的权重
#         weight = weight[target.long()]  # 使用 target 的值来选择相应的类别权重
#         loss = weight * (target * torch.log(output)) + (1 - target) * torch.log(1 - output)
#     else:
#         loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
#     return -torch.mean(loss)

# def evaluate(model, test_loader, device):
#     model.eval()  # 将模型设置为评估模式
#     preds = []
#     labels = []
    
#     with torch.no_grad():  # 禁用梯度计算，提高推理效率
#         for data in test_loader:
#             data.to(device)
#             pred = model(data.x, data.edge_index, data.edge_type, data.batch)
#             preds.append(pred.detach().cpu().numpy())  # 获取预测结果
#             labels.append(data.y.detach().cpu().numpy())  # 获取实际标签

#     preds = np.concatenate(preds, axis=0)
#     labels = np.concatenate(labels, axis=0)
    
#     # 计算评估指标，例如准确率
#     accuracy = accuracy_score(labels, (preds > 0.5).astype(int))
#     f1 = f1_score(labels, (preds > 0.5).astype(int))
#     # auc = roc_auc_score(labels, preds)
#     roc_auc_value = roc_auc_score(labels, preds)  # 计算 AUC

#     print(f'Accuracy: {accuracy:.4f}')
#     print(f'F1 Score: {f1:.4f}')
#     # print(f'AUC: {auc:.4f}')
#     print(f'AUC: {roc_auc_value:.4f}')


#         # 绘制 ROC 曲线
#     fpr, tpr, _ = roc_curve(labels, preds)
#     roc_auc_score_value = auc(fpr, tpr)  # 避免与 sklearn.metrics.auc 产生冲突
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score_value:.4f})")
#     # roc_auc = auc(fpr, tpr)
#     # plt.figure(figsize=(8, 6))
#     # plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
#     plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Receiver Operating Characteristic (ROC)")
#     plt.legend(loc="lower right")
#     plt.show()

#     accuracy = accuracy_score(labels, (preds > 0.5).astype(int))
#     f1 = f1_score(labels, (preds > 0.5).astype(int))
#     print(f'Accuracy: {accuracy:.4f}')
#     print(f'F1 Score: {f1:.4f}')


# # # 定义 weighted_BCE
# # def weighted_BCE(output, target, weight=None):  
# #     if weight is not None:     
# #         loss = weight * (target * torch.log(output)) + (1 - target) * torch.log(1 - output)
# #     else:
# #         loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
# #     return -torch.mean(loss)

# # 数据预处理函数 preprocess
# def preprocess():
#     global rootpath
#     rootpath = 'D:/CC/PycharmProjects/GoGT/JAK_ML/jak/'
#     JAK = [rootpath+'final_JAK2(2).csv']
#     LABEL = [rootpath+'JAK2_graph_labels.txt']
#     nodes = []
#     edges = []
#     relations = []
#     y = []
#     smiles_all = []

#     for path1, path2 in zip(JAK, LABEL):
#         smiles = pd.read_csv(path1)['Smiles'].tolist()
#         y_ = ((np.array(pd.read_csv(path1)['Activity'].tolist())>0)*1).tolist()
#         smiles_all.extend(smiles)

#         lens = []
#         adjs = []
#         ords = []
#         for i in range(len(smiles)):
#             node, adj, order = gen_smiles2graph(smiles[i])
#             if node is None or adj is None:  # 如果解析失败，则跳过此样本
#                 continue
#         # # 修改判断条件，避免与字符串进行不正确的比较
#         #     if isinstance(node, str) and node == 'error':
#         #         continue  # 如果返回 'error' 字符串，则跳过该分子
#             # if node == 'error':
#             #     continue
#             lens.append(adj.shape[0])
#             adjs.append(adj)  
#             ords.append(order)
#             node[:,2] += 1
#             node[:,3] -= 1
#             nodes.append(node)
        
#         # adjs = np.array(adjs)
#                 # 在将 adjs 转换为 NumPy 数组之前，检查每个元素的形状
#         if len(adjs) > 0 and all(adj.shape == adjs[0].shape for adj in adjs):  # 确保所有矩阵形状一致
#             adjs = np.array(adjs)
#         else:
#             print("Warning: The shapes of the adjacency matrices are not consistent.")
#             adjs = np.array(adjs, dtype=object)  # 如果形状不一致，可以转换为 dtype=object
#         lens = np.array(lens)
    
#         def adj2idx(adj):
#             idx = []
#             for i in range(adj.shape[0]):
#                 for j in range(adj.shape[1]):
#                     if adj[i,j] == 1:
#                         idx.append([i,j])
#             return np.array(idx)
        
#         def order2relation(adj):
#             idx = []
#             for i in range(adj.shape[0]):
#                 for j in range(adj.shape[1]):
#                     if adj[i,j] != 0:
#                         idx.extend([adj[i,j]])
#             return np.array(idx)
        
#         for i in range(lens.shape[0]):
#             adj = adjs[i]
#             order = ords[i]
#             idx = adj2idx(adj)
#             relation = order2relation(order)-1
#             edges.append(idx)
#             relations.append(relation)
#             y.append(y_[i])  # 确保标签添加与 edges 一致
#                 # 确保 labels 和 edges, nodes, relations 长度一致
#     assert len(edges) == len(nodes) == len(relations) == len(y), "Lengths are not consistent!"
#     #     y.append(np.array(y_))
#     # y = np.concatenate(y)
#     y = np.array(y) 
#         # 在此处添加检查
#     print(f"Edges length: {len(edges)}")
#     print(f"Nodes length: {len(nodes)}")
#     print(f"Relations length: {len(relations)}")
#     print(f"Labels length: {len(y)}")
    
#     return smiles_all, nodes, edges, relations, y

# # 数据集类 GDataset
# class GDataset(Dataset):
#     def __init__(self, nodes, edges, relations, y, idx):
#         super(GDataset, self).__init__()
#         self.nodes = nodes
#         self.edges = edges
#         self.y = y
#         self.relations = relations
#         self.idx = idx
        
#     def __getitem__(self, idx):
#         # if idx >= len(self.edges):  # 如果索引超出了边的数量，报错
#         #     raise IndexError(f"Index {idx} is out of range for edges of size {len(self.edges)}")
#         # print(f"Accessing index {idx}, total edges: {len(self.edges)}")


#         idx = self.idx[idx]
#         edge_index = torch.tensor(self.edges[idx].T, dtype=torch.long)
#         x = torch.tensor(self.nodes[idx], dtype=torch.long)
#         y = torch.tensor(self.y[idx], dtype=torch.float)
#         edge_type = torch.tensor(self.relations[idx], dtype=torch.float)
#         return Data(x=x,edge_index=edge_index,edge_type=edge_type,y=y)

    
#     def __len__(self):
#         return len(self.idx)
    
#   # 模型定义 RGCN  
# class RGCN(torch.nn.Module):
#     def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
#         super(RGCN, self).__init__()
#         # 根据数据动态调整每列的嵌入层大小
#         self.max_values = [27, 6, 2, 6, 1, 2]  # 对应每列的最大值
#         # max_values = [27, 4, 2, 3, 1, 2]   # 对应每列的最大值
#         self.embedding = nn.ModuleList([
#             nn.Embedding(max_value + 1, in_embd)  # 每个嵌入层的大小至少是 max_value + 1
#             for max_value in self.max_values
#         ])
#         # self.embedding = nn.ModuleList([nn.Embedding(33,in_embd), nn.Embedding(5,in_embd), \
#         #                   nn.Embedding(3,in_embd), nn.Embedding(4,in_embd), \
#         #                   nn.Embedding(2,in_embd), nn.Embedding(3,in_embd)])
        
#         self.RGCNConv1 = RGCNConv(6*in_embd, layer_embd, num_relations)
#         self.RGCNConv2 = RGCNConv(layer_embd, out_embd, num_relations)
        
#         self.activation = nn.Sigmoid()
#         self.pool = GlobalAttention(gate_nn=nn.Sequential(
#             nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
#         ))
        
#         self.graph_linear = nn.Linear(out_embd, 1)

#     def forward(self, x, edge_index, edge_type, batch):
#         for i in range(6):
#             if x[:, i].max() >= (self.max_values[i] + 1):
#                 print(f"Warning: index out of range for column {i}, max index: {x[:, i].max()}")
#             embds = self.embedding[i](x[:, i])
#             if i == 0:
#                 x_ = embds
#             else:
#                 x_ = torch.cat((x_, embds), 1)
#         edge_type = edge_type.long()
#         out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
#         out = self.activation(self.RGCNConv2(out, edge_index, edge_type))
#         out = self.pool(out, batch)
#         out = self.graph_linear(out)
#         return self.activation(out)



# def train(model_name, EPOCH, split, batch_size, weight):
#     smiles, nodes, edges, relations, y = preprocess()

#     # 计算训练集和测试集的大小
#     train_num = int(split * len(smiles))  # 按照比例计算训练集大小
#     test_num = len(smiles) - train_num    # 剩余部分为测试集大小

#     # 数据集划分
#     train_set, test_set = torch.utils.data.random_split(
#         GDataset(nodes, edges, relations, y, range(len(smiles))),
#         [train_num, test_num],
#         generator=torch.Generator().manual_seed(42)
#     )

#     print(f"Train set size: {len(train_set)}")
#     print(f"Test set size: {len(test_set)}")

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


#     # 选择模型
#     if model_name == 'RGCN':
#         model = RGCN(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
#     # elif model_name == 'GCN':
#     #     model = GCN(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
#     # elif model_name == 'GAT':
#     #     model = GAT(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)

#     model.apply(weights_init)  # 权重初始化
#     model.to(device)  # 将模型移动到正确的设备

#     # 定义优化器
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
#     train_losses = []
#     test_losses = []

#     for epoch in range(EPOCH):
#         total_loss = 0  # 初始化总损失值
#         print(f"Epoch {epoch + 1}/{EPOCH}:")
#         model.train()  # 设置模型为训练模式

#         # print('epoch {}:'.format(epoch + 1))
#         for data in train_loader:
#             data.to(device)
#             pred = model(data.x, data.edge_index, data.edge_type, data.batch)
#                         # 传递类别权重
#             weight_tensor = torch.FloatTensor([weight, 1]).to(device)  # 转移到正确的设备
#             loss = weighted_BCE(pred.squeeze(), data.y, weight_tensor)
#             # loss = weighted_BCE(pred.squeeze(), data.y, torch.FloatTensor([weight, 1]))
            

#             optimizer.zero_grad()  # 清除之前的梯度
#             loss.backward()  # 反向传播
#             optimizer.step()  # 更新模型参数

#             total_loss += loss.item() * data.y.size(0)  # 累加损失并加权

#         avg_train_loss = total_loss / len(train_loader.dataset)  # 计算平均损失
#         train_losses.append(avg_train_loss)
#         print(f"Train Loss: {avg_train_loss:.4f}")  # 打印训练损失
        
#     # # 每个 epoch 结束后评估模型在测试集上的性能
#     #     evaluate(model, test_loader, device)

#         # 评估测试集损失
#         model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for data in test_loader:
#                 data.to(device)
#                 pred = model(data.x, data.edge_index, data.edge_type, data.batch)
#                 loss = weighted_BCE(pred.squeeze(), data.y, weight_tensor)
#                 test_loss += loss.item() * data.y.size(0)
#         avg_test_loss = test_loss / len(test_loader.dataset)
#         test_losses.append(avg_test_loss)

#         print(f"Epoch {epoch + 1}/{EPOCH}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

#     # 绘制损失曲线
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, EPOCH + 1), train_losses, label="Train Loss")
#     plt.plot(range(1, EPOCH + 1), test_losses, label="Test Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Testing Loss")
#     plt.legend()
#     plt.show()

#     # **在训练结束后调用 evaluate() 进行模型评估**
#     print("\nEvaluating Model on Test Data...\n")
#     evaluate(model, test_loader, device)  # **调用 evaluate() 进行测试，并绘制 ROC 曲线**

#     # 保存模型
#     torch.save(model.state_dict(), 'GVAE_JAK2.pt')


# # 开始训练
# train('RGCN', EPOCH=50, split=0.8, batch_size=32, weight=2)






##统一要求的RGCN\GAT\GCN模型
##不含模型保存
# import numpy as np
# import pandas as pd
# import rdkit
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import DataLoader, Data
# from torch_geometric.nn import GATConv, RGCNConv, GCNConv, GlobalAttention
# from sklearn.model_selection import StratifiedKFold, ParameterGrid
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# import pickle
# from itertools import cycle
# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from torch_geometric.data import Dataset
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # 定义设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 定义权重初始化函数
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)  # 偏置初始化为 0
#     elif isinstance(m, nn.Embedding):
#         nn.init.xavier_uniform_(m.weight)  # Embedding 层也使用 Xavier 初始化

# # 定义 GDataset 类
# class GDataset(Dataset):
#     def __init__(self, nodes, edges, relations, y, idx):
#         super(GDataset, self).__init__()
#         self.nodes = nodes
#         self.edges = edges
#         self.relations = relations
#         self.y = y
#         self.idx = idx

#     def __len__(self):
#         return len(self.idx)

#     def __getitem__(self, idx):
#         idx = self.idx[idx]
#         edge_index = torch.tensor(self.edges[idx].T, dtype=torch.long)  # 边索引
#         x = torch.tensor(self.nodes[idx], dtype=torch.long)  # 节点特征
#         y = torch.tensor(self.y[idx], dtype=torch.float)  # 标签
#         edge_type = torch.tensor(self.relations[idx], dtype=torch.float)  # 边类型

#         embedding_sizes = [33, 5, 3, 4, 2, 3]
#         for i in range(6):
#             x[:, i] = torch.clamp(x[:, i], 0, embedding_sizes[i] - 1)  # 修正无效索引

#         return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)

# # 定义数据预处理函数

# def preprocess(dataset_name):
#     rootpath = 'D:/CC/PycharmProjects/GoGT/JAK_ML/data/'  # 修改为你的数据路径
#     JAK = [rootpath + f'unique_Homo_um_pact_act{dataset_name}.xlsx']  # 更改为xlsx扩展名
#     LABEL = [rootpath + f'{dataset_name}_graph_labels.txt']
#     nodes = []
#     edges = []
#     relations = []
#     y = []
#     smiles_all = []

#     for path1, path2 in zip(JAK, LABEL):
#         # 修改为读取xlsx文件
#         df = pd.read_excel(path1)
#         smiles = df['canonical_smiles'].tolist()
#         y_ = ((np.array(df['Activity'].tolist()) > 0) * 1)
#         smiles_all.extend(smiles)

#         lens = []
#         adjs = []  # 保持为 Python 列表
#         ords = []
#         for i in range(len(smiles)):
#             node, adj, order = gen_smiles2graph(smiles[i])
#             if node is None or adj is None:  # 如果解析失败，则跳过此样本
#                 continue
#             lens.append(adj.shape[0])
#             adjs.append(adj)  # 直接添加 adj，不转换为 NumPy 数组
#             ords.append(order)
#             node[:, 2] += 1
#             node[:, 3] -= 1
#             nodes.append(node)

#         lens = np.array(lens)

#         def adj2idx(adj):
#             idx = []
#             for i in range(adj.shape[0]):
#                 for j in range(adj.shape[1]):
#                     if adj[i, j] == 1:
#                         idx.append([i, j])
#             return np.array(idx)

#         def order2relation(adj):
#             idx = []
#             for i in range(adj.shape[0]):
#                 for j in range(adj.shape[1]):
#                     if adj[i, j] != 0:
#                         idx.extend([adj[i, j]])
#             return np.array(idx)

#         for i in range(lens.shape[0]):
#             adj = adjs[i]  # 直接从列表中获取 adj
#             order = ords[i]
#             idx = adj2idx(adj)
#             relation = order2relation(order) - 1
#             edges.append(idx)
#             relations.append(relation)
#             y.append(y_[i])  # 确保标签添加与 edges 一致

#     y = np.array(y)
#     return smiles_all, nodes, edges, relations, y
# # 定义 gen_smiles2graph 函数
# def gen_smiles2graph(sml):
#     ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]
#     dic = {ls[i]: i for i in range(len(ls))}

#     m = rdkit.Chem.MolFromSmiles(sml)
#     if not m:
#         print(f"Error parsing SMILES: {sml}")
#         return None, None, None

#     order_string = {
#         rdkit.Chem.rdchem.BondType.SINGLE: 1,
#         rdkit.Chem.rdchem.BondType.DOUBLE: 2,
#         rdkit.Chem.rdchem.BondType.TRIPLE: 3,
#         rdkit.Chem.rdchem.BondType.AROMATIC: 4,
#     }

#     N = len(list(m.GetAtoms()))
#     nodes = np.zeros((N, 6))

#     for i in m.GetAtoms():
#         atom_types = dic.get(i.GetAtomicNum(), 0)
#         degree = min(i.GetDegree(), 4)  # 确保度不超过 4
#         nodes[i.GetIdx()] = [
#             atom_types, degree, i.GetFormalCharge(),
#             i.GetHybridization(), i.GetIsAromatic(), i.GetChiralTag()
#         ]

#     adj = np.zeros((N, N))
#     orders = np.zeros((N, N))
#     for bond in m.GetBonds():
#         u = bond.GetBeginAtomIdx()
#         v = bond.GetEndAtomIdx()
#         bond_type = bond.GetBondType()
#         adj[u, v] = adj[v, u] = 1
#         orders[u, v] = orders[v, u] = order_string.get(bond_type, 0)

#     return nodes, adj, orders

# # 定义模型类（RGCN、GCN、GAT）
# class RGCN(nn.Module):
#     def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
#         super(RGCN, self).__init__()
#         self.embedding = nn.ModuleList([nn.Embedding(33, in_embd), nn.Embedding(5, in_embd),
#                           nn.Embedding(3, in_embd), nn.Embedding(4, in_embd),
#                           nn.Embedding(2, in_embd), nn.Embedding(3, in_embd)])
#         self.RGCNConv1 = RGCNConv(6 * in_embd, layer_embd, num_relations)
#         self.RGCNConv2 = RGCNConv(layer_embd, out_embd, num_relations)
#         self.activation = nn.Sigmoid()
#         self.pool = GlobalAttention(gate_nn=nn.Sequential(
#             nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
#         ))
#         self.graph_linear = nn.Linear(out_embd, 1)

#     def forward(self, x, edge_index, edge_type, batch):
#         # 检查输入数据的维度
#         if x.dim() != 2:
#             print(f"Invalid node feature dimension: {x.dim()}")
#             raise ValueError("Invalid node feature dimension.")
#         if edge_index.dim() != 2 or edge_index.size(0) != 2:
#             print(f"Invalid edge index dimension: {edge_index.dim()}")
#             raise ValueError("Invalid edge index dimension.")
        
#         for i in range(6):
#             embds = self.embedding[i](x[:, i])
#             if i == 0:
#                 x_ = embds
#             else:
#                 x_ = torch.cat((x_, embds), 1)
#         edge_type = edge_type.long()
#         out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
#         out = self.activation(self.RGCNConv2(out, edge_index, edge_type))
#         out = self.pool(out, batch)
#         out = self.graph_linear(out)
#         return self.activation(out)

# class GCN(nn.Module):
#     def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
#         super(GCN, self).__init__()
#         self.embedding = nn.ModuleList([nn.Embedding(33, in_embd), nn.Embedding(5, in_embd),
#                           nn.Embedding(3, in_embd), nn.Embedding(4, in_embd),
#                           nn.Embedding(2, in_embd), nn.Embedding(3, in_embd)])
#         self.GCNConv1 = GCNConv(6 * in_embd, layer_embd)
#         self.GCNConv2 = GCNConv(layer_embd, out_embd)
#         self.activation = nn.Sigmoid()
#         self.pool = GlobalAttention(gate_nn=nn.Sequential(
#             nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
#         ))
#         self.graph_linear = nn.Linear(out_embd, 1)

#     def forward(self, x, edge_index, edge_type, batch):
#         # 检查输入数据的维度
#         if x.dim() != 2:
#             print(f"Invalid node feature dimension: {x.dim()}")
#             raise ValueError("Invalid node feature dimension.")
#         if edge_index.dim() != 2 or edge_index.size(0) != 2:
#             print(f"Invalid edge index dimension: {edge_index.dim()}")
#             raise ValueError("Invalid edge index dimension.")
        
#         for i in range(6):
#             embds = self.embedding[i](x[:, i])
#             if i == 0:
#                 x_ = embds
#             else:
#                 x_ = torch.cat((x_, embds), 1)
#         out = self.activation(self.GCNConv1(x_, edge_index))
#         out = self.activation(self.GCNConv2(out, edge_index))
#         out = self.pool(out, batch)
#         out = self.graph_linear(out)
#         return self.activation(out)

# class GAT(nn.Module):
#     def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
#         super(GAT, self).__init__()
#         self.embedding = nn.ModuleList([nn.Embedding(33, in_embd), nn.Embedding(5, in_embd),
#                           nn.Embedding(3, in_embd), nn.Embedding(4, in_embd),
#                           nn.Embedding(2, in_embd), nn.Embedding(3, in_embd)])
#         self.GATConv1 = GATConv(6 * in_embd, layer_embd, heads=2, concat=False)
#         self.GATConv2 = GATConv(layer_embd, out_embd, heads=2, concat=False)
#         self.activation = nn.Sigmoid()
#         self.pool = GlobalAttention(gate_nn=nn.Sequential(
#             nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
#         ))
#         self.graph_linear = nn.Linear(out_embd, 1)

#     def forward(self, x, edge_index, edge_type, batch):
#         # 检查输入数据的维度
#         if x.dim() != 2:
#             print(f"Invalid node feature dimension: {x.dim()}")
#             raise ValueError("Invalid node feature dimension.")
#         if edge_index.dim() != 2 or edge_index.size(0) != 2:
#             print(f"Invalid edge index dimension: {edge_index.dim()}")
#             raise ValueError("Invalid edge index dimension.")

#         for i in range(6):
#             embds = self.embedding[i](x[:, i])
#             if i == 0:
#                 x_ = embds
#             else:
#                 x_ = torch.cat((x_, embds), 1)
#         out = self.activation(self.GATConv1(x_, edge_index))
#         out = self.activation(self.GATConv2(out, edge_index))
#         out = self.pool(out, batch)
#         out = self.graph_linear(out)
#         return self.activation(out)

# # 定义评估函数
# def evaluate(model, test_loader, device):
#     model.eval()
#     y_true = []
#     y_pred = []
#     y_score = []

#     with torch.no_grad():
#         for data in test_loader:
#             data.to(device)
#             # 检查数据维度和类型
#             if data.x.dim() != 2:
#                 print(f"Invalid node feature dimension in evaluation: {data.x.dim()}")
#                 raise ValueError("Invalid node feature dimension in evaluation.")
#             if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
#                 print(f"Invalid edge index dimension in evaluation: {data.edge_index.dim()}")
#                 raise ValueError("Invalid edge index dimension in evaluation.")
#             pred = model(data.x, data.edge_index, data.edge_type, data.batch)
#             y_true.extend(data.y.cpu().numpy())
#             y_pred.extend((pred > 0.5).float().cpu().numpy())
#             y_score.extend(pred.cpu().numpy())

#     # 计算评估指标
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_score)

#     # 输出分类报告和混淆矩阵
#     print("Classification Report:")
#     print(classification_report(y_true, y_pred))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))

#     # 计算 ROC 曲线
#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     roc_auc = auc(fpr, tpr)

#     return accuracy, precision, recall, f1, roc_auc, fpr, tpr

# # 定义训练函数
# def train_single_fold(model, train_loader, test_loader, optimizer, scheduler, EPOCH, weight):
#     model.to(device)
#     train_losses = []
#     test_losses = []

#     for epoch in range(EPOCH):
#         model.train()
#         total_loss = 0

#         for data in train_loader:
#             data.to(device)
#             # 检查数据维度和类型
#             if data.x.dim() != 2:
#                 print(f"Invalid node feature dimension in training: {data.x.dim()}")
#                 raise ValueError("Invalid node feature dimension in training.")
#             if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
#                 print(f"Invalid edge index dimension in training: {data.edge_index.dim()}")
#                 raise ValueError("Invalid edge index dimension in training.")
#             pred = model(data.x, data.edge_index, data.edge_type, data.batch)

#             # 动态选择类别权重
#             weight_tensor = torch.where(data.y == 1, torch.tensor(weight, device=device), torch.tensor(1.0, device=device))
#             loss = F.binary_cross_entropy(pred.squeeze(), data.y, weight=weight_tensor)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * data.y.size(0)

#         avg_train_loss = total_loss / len(train_loader.dataset)
#         train_losses.append(avg_train_loss)
#         torch.cuda.empty_cache()  # 清空缓存
#         # 评估测试集
#         model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for data in test_loader:
#                 data.to(device)
#                 # 检查数据维度和类型
#                 if data.x.dim() != 2:
#                     print(f"Invalid node feature dimension in testing: {data.x.dim()}")
#                     raise ValueError("Invalid node feature dimension in testing.")
#                 if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
#                     print(f"Invalid edge index dimension in testing: {data.edge_index.dim()}")
#                     raise ValueError("Invalid edge index dimension in testing.")
#                 pred = model(data.x, data.edge_index, data.edge_type, data.batch)

#                 # 动态选择类别权重
#                 weight_tensor = torch.where(data.y == 1, torch.tensor(weight, device=device), torch.tensor(1.0, device=device))
#                 loss = F.binary_cross_entropy(pred.squeeze(), data.y, weight=weight_tensor)

#                 test_loss += loss.item() * data.y.size(0)

#         avg_test_loss = test_loss / len(test_loader.dataset)
#         test_losses.append(avg_test_loss)

#         # 更新学习率
#         scheduler.step(avg_test_loss)

#         print(f"Epoch {epoch + 1}/{EPOCH}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

#     return {'train_losses': train_losses, 'test_losses': test_losses}

# # 定义网格调优函数
# def grid_search(model_name, dataset_name, param_grid, EPOCH, batch_size, weight, n_splits=5):
#     smiles, nodes, edges, relations, y = preprocess(dataset_name)

#     # 使用 StratifiedKFold 进行五折交叉验证
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     # results = []

#     # 记录当前模型的最佳 ROC 数据
#     best_roc_auc = 0
#     best_fpr = None
#     best_tpr = None
#     best_metrics = None
#     for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
#         print(f"\nFold {fold + 1}/{n_splits}")

#         # 划分训练集和测试集
#         train_set = GDataset(nodes, edges, relations, y, train_idx)
#         test_set = GDataset(nodes, edges, relations, y, test_idx)

#         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#         # 网格调优
#         for params in ParameterGrid(param_grid):
#             print(f"Testing parameters: {params}")

#             # 初始化模型
#             if model_name == 'RGCN':
#                 model = RGCN(in_embd=params['in_embd'], layer_embd=params['layer_embd'],
#                              out_embd=params['out_embd'], num_relations=4, dropout=params['dropout'])
#             elif model_name == 'GCN':
#                 model = GCN(in_embd=params['in_embd'], layer_embd=params['layer_embd'],
#                             out_embd=params['out_embd'], num_relations=4, dropout=params['dropout'])
#             elif model_name == 'GAT':
#                 model = GAT(in_embd=params['in_embd'], layer_embd=params['layer_embd'],
#                             out_embd=params['out_embd'], num_relations=4, dropout=params['dropout'])
#             model.apply(weights_init)
#             model.to(device)

#             # 定义优化器和学习率调度器
#             optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
#             scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

#             # 训练模型
#             result = train_single_fold(model, train_loader, test_loader, optimizer, scheduler, EPOCH, weight)

#             # 评估模型
#             accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate(model, test_loader, device)
#             metrics = {
#                 'accuracy': accuracy,
#                 'precision': precision,
#                 'recall': recall,
#                 'f1': f1,
#                 'roc_auc': roc_auc
#             }

#             if roc_auc > best_roc_auc:
#                 best_roc_auc = roc_auc
#                 best_fpr = fpr
#                 best_tpr = tpr
#                 best_metrics = metrics

#     return {
#         'fpr': best_fpr,
#         'tpr': best_tpr,
#         'roc_auc': best_roc_auc,
#         'metrics': best_metrics
#     }

# # 定义超参数网格
# param_grid = {
#     'in_embd': [8],
#     'layer_embd': [32],
#     'out_embd': [64],
#     'dropout': [0.0],
#     'lr': [1e-3]
# }

# # 定义数据集列表
# datasets = ['JAK1', 'JAK2', 'JAK3', 'TYK2']

# # 定义模型列表、训练轮数和批量大小
# models = ['RGCN', 'GCN', 'GAT']
# epochs = [2, 2, 2]
# batch_sizes = [64, 64, 64]

# # 保存所有 ROC 数据和最佳指标
# all_roc_data = {}

# # 运行网格搜索
# for dataset_name in datasets:
#     all_roc_data[dataset_name] = {}
#     for model_name, epoch, batch_size in zip(models, epochs, batch_sizes):
#         print(f"Running grid search for {model_name} on {dataset_name}...")
#         roc_data = grid_search(model_name, dataset_name, param_grid, EPOCH=epoch, batch_size=batch_size, weight=2)
#         all_roc_data[dataset_name][model_name] = roc_data

# # 定义颜色和线型
# colors = cycle(['darkorange', 'blue', 'green', 'red'])
# linestyles = cycle(['-', '--', '-.', ':'])

# # 保存所有最佳评估指标
# all_best_metrics = {}

# # 遍历每个模型，分别绘制其在不同数据集上的 ROC 曲线
# for model_name in models:
#     plt.figure(figsize=(10, 8))
#     best_metrics = {}
#     for dataset_name in datasets:
#         roc_data = all_roc_data[dataset_name][model_name]
#         fpr = roc_data['fpr']
#         tpr = roc_data['tpr']
#         roc_auc = roc_data['roc_auc']

#         # 绘制 ROC 曲线
#         plt.plot(fpr, tpr, color=next(colors), linestyle=next(linestyles),
#                  label=f'{dataset_name} (AUC = {roc_auc:.2f})')

#         # 获取最佳评估指标
#         best_metrics[dataset_name] = roc_data['metrics']

#     # 绘制随机猜测线
#     plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

#     # 设置图表属性
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curves for {model_name} on JAK1, JAK2, JAK3, and TYK2 Datasets')
#     plt.legend(loc="lower right")
#     plt.savefig(f'{model_name}_roc_curves.png')
#     plt.show()

#     all_best_metrics[model_name] = best_metrics

# # 打印每个模型在每个数据集上的最佳评估指标
# for model_name, metrics in all_best_metrics.items():
#     print(f"\nBest evaluation metrics for {model_name}:")
#     for dataset_name, metric_values in metrics.items():
#         print(f"  Dataset: {dataset_name}")
#         print(f"    Accuracy: {metric_values['accuracy']:.4f}")
#         print(f"    Precision: {metric_values['precision']:.4f}")
#         print(f"    Recall: {metric_values['recall']:.4f}")
#         print(f"    F1-score: {metric_values['f1']:.4f}")
#         print(f"    ROC AUC: {metric_values['roc_auc']:.4f}")





###统一好的rgcn、gcn、gat的完整代码
import numpy as np
import pandas as pd
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
from collections import defaultdict
import random  # 新增导入
import time    # 新增导入
import inspect # 新增导入
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# === 新增函数：设置随机种子 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在训练开始前设置种子
set_seed(42)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义权重初始化函数
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 偏置初始化为 0
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)  # Embedding 层也使用 Xavier 初始化

# 定义 GDataset 类
class GDataset(Dataset):
    def __init__(self, nodes, edges, relations, y, idx, smiles_all):  # 添加smiles_all参数
        super(GDataset, self).__init__()
        self.nodes = nodes
        self.edges = edges
        self.relations = relations
        self.y = y
        self.idx = idx
        self.smiles_all = smiles_all  # 保存SMILES列表

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        data_idx = self.idx[idx]
        edge_index = torch.tensor(self.edges[data_idx].T, dtype=torch.long)
        x = torch.tensor(self.nodes[data_idx], dtype=torch.long)
        y = torch.tensor(self.y[data_idx], dtype=torch.float)
        edge_type = torch.tensor(self.relations[data_idx], dtype=torch.float)

        embedding_sizes = [33, 5, 3, 4, 2, 3]
        for i in range(6):
            x[:, i] = torch.clamp(x[:, i], 0, embedding_sizes[i] - 1)  # 修正无效索引

        # 添加SMILES到Data对象
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
        data.smiles = self.smiles_all[data_idx]  # 添加SMILES信息
        return data

# 定义数据预处理函数
def preprocess(dataset_name):
    rootpath = 'D:/CC/PycharmProjects/GoGT/JAK_ML/data/'  # 修改为你的数据路径
    JAK = [rootpath + f'unique_Homo_um_pact_act{dataset_name}.xlsx']  # 修改为xlsx
    LABEL = [rootpath + f'{dataset_name}_graph_labels.txt']
    nodes = []
    edges = []
    relations = []
    y = []
    smiles_all = []

    for path1, path2 in zip(JAK, LABEL):
        # 修改为读取xlsx文件
        smiles = pd.read_excel(path1)['canonical_smiles'].tolist()
        y_ = ((np.array(pd.read_excel(path1)['Activity'].tolist()) > 0) * 1)
        smiles_all.extend(smiles)

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

# 定义模型类（RGCN、GCN、GAT）
class RGCN(nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(RGCN, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(33, in_embd), nn.Embedding(5, in_embd),
                          nn.Embedding(3, in_embd), nn.Embedding(4, in_embd),
                          nn.Embedding(2, in_embd), nn.Embedding(3, in_embd)])
        self.RGCNConv1 = RGCNConv(6 * in_embd, layer_embd, num_relations)
        self.RGCNConv2 = RGCNConv(layer_embd, out_embd, num_relations)
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
        ))
        self.graph_linear = nn.Linear(out_embd, 1)

    def forward(self, x, edge_index, edge_type, batch):
        # 检查输入数据的维度
        if x.dim() != 2:
            print(f"Invalid node feature dimension: {x.dim()}")
            raise ValueError("Invalid node feature dimension.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            print(f"Invalid edge index dimension: {edge_index.dim()}")
            raise ValueError("Invalid edge index dimension.")
        
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
        edge_type = edge_type.long()
        out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
        out = self.activation(self.RGCNConv2(out, edge_index, edge_type))
        out = self.pool(out, batch)
        out = self.graph_linear(out)
        return self.activation(out)

class GCN(nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GCN, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(33, in_embd), nn.Embedding(5, in_embd),
                          nn.Embedding(3, in_embd), nn.Embedding(4, in_embd),
                          nn.Embedding(2, in_embd), nn.Embedding(3, in_embd)])
        self.GCNConv1 = GCNConv(6 * in_embd, layer_embd)
        self.GCNConv2 = GCNConv(layer_embd, out_embd)
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
        ))
        self.graph_linear = nn.Linear(out_embd, 1)
        self.num_relations = num_relations  # 虽然不使用但保持参数一致
    def forward(self, x, edge_index, edge_type, batch):
        # 检查输入数据的维度
        if x.dim() != 2:
            print(f"Invalid node feature dimension: {x.dim()}")
            raise ValueError("Invalid node feature dimension.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            print(f"Invalid edge index dimension: {edge_index.dim()}")
            raise ValueError("Invalid edge index dimension.")
        
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
        out = self.activation(self.GCNConv1(x_, edge_index))
        out = self.activation(self.GCNConv2(out, edge_index))
        out = self.pool(out, batch)
        out = self.graph_linear(out)
        return self.activation(out)

# === 修改后的模型类（仅GAT需要修改）===
class GAT(nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout  # 新增保存dropout率
        self.embedding = nn.ModuleList([nn.Embedding(33, in_embd), nn.Embedding(5, in_embd),
                          nn.Embedding(3, in_embd), nn.Embedding(4, in_embd),
                          nn.Embedding(2, in_embd), nn.Embedding(3, in_embd)])
        self.GATConv1 = GATConv(6 * in_embd, layer_embd, heads=2, concat=False, dropout=dropout)
        self.GATConv2 = GATConv(layer_embd, out_embd, heads=2, concat=False, dropout=dropout)
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)
        ))
        self.graph_linear = nn.Linear(out_embd, 1)

    def forward(self, x, edge_index, edge_type, batch):
        # 新增：预测时禁用dropout
        if not self.training:
            self.GATConv1.dropout = 0.0
            self.GATConv2.dropout = 0.0

        # 原有forward逻辑保持不变
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))
        out = self.pool(out, batch)
        out = self.graph_linear(out)
        
        # 新增：恢复dropout设置（如果是训练模式）
        if self.training:
            self.GATConv1.dropout = self.dropout
            self.GATConv2.dropout = self.dropout
            
        return torch.sigmoid(out)

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
            pred = model(data.x, data.edge_index, data.edge_type, data.batch)
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

# 定义训练函数
def train_single_fold(model, train_loader, test_loader, optimizer, scheduler, EPOCH, weight):
    model.to(device)
    train_losses = []
    test_losses = []
    val_predictions = []  # 新增：记录验证预测

    for epoch in range(EPOCH):
        model.train()
        total_loss = 0

        for data in train_loader:
            data.to(device)
            if data.x.dim() != 2:
                print(f"Invalid node feature dimension in training: {data.x.dim()}")
                raise ValueError("Invalid node feature dimension in training.")
            if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
                print(f"Invalid edge index dimension in training: {data.edge_index.dim()}")
                raise ValueError("Invalid edge index dimension in training.")
            pred = model(data.x, data.edge_index, data.edge_type, data.batch)
            weight_tensor = torch.where(data.y == 1, torch.tensor(weight, device=device), torch.tensor(1.0, device=device))
            loss = F.binary_cross_entropy(pred.squeeze(), data.y, weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.y.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        torch.cuda.empty_cache()

        # 验证阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data.to(device)
                pred = model(data.x, data.edge_index, data.edge_type, data.batch)
                # 新增：记录验证预测
                val_predictions.append({
                    'smiles': data.smiles,  # 假设data中有smiles属性
                    'prediction': pred.cpu().numpy(),
                    'true_label': data.y.cpu().numpy(),
                    'epoch': epoch
                })
                # 原有验证逻辑...
                weight_tensor = torch.where(data.y == 1, torch.tensor(weight, device=device), torch.tensor(1.0, device=device))
                loss = F.binary_cross_entropy(pred.squeeze(), data.y, weight=weight_tensor)
                test_loss += loss.item() * data.y.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        scheduler.step(avg_test_loss)

        print(f"Epoch {epoch + 1}/{EPOCH}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

    # 新增：保存验证预测
    torch.save(val_predictions, f"val_predictions_{model.__class__.__name__}.pt")
    return {'train_losses': train_losses, 'test_losses': test_losses}

# === 修改后的模型保存函数 ===
def save_model(model, optimizer, model_name, dataset_name, params, metrics, epoch):
    try:
        save_dir = f"./saved_models/{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 新增：保存完整配置
        save_content = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'params': params,
            'metrics': metrics,
            'epoch': epoch,
            'model_type': model_name,
            'config': {
                'atom_types': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 
                             16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 
                             52, 53, 54, 55, 56, 83, 88],
                'bond_type_mapping': {
                    'SINGLE': 1,
                    'DOUBLE': 2,
                    'TRIPLE': 3,
                    'AROMATIC': 4
                },
                'random_seed': 42,
                'device': str(device),
            },
            'model_architecture': inspect.getsource(model.__class__)  # 保存类定义
        }
        
        save_path = f"{save_dir}/{model_name}_best.pth"
        torch.save(save_content, save_path)
        print(f"Model saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save model: {str(e)}")
        return False

# === grid_search函数修改 ===
def grid_search(model_name, dataset_name, param_grid, EPOCH, batch_size, weight, n_splits=5):
    try:
        smiles, nodes, edges, relations, y = preprocess(dataset_name)
        if len(y) == 0:
            print(f"No valid data for {dataset_name}")
            return {
                'fpr': None,
                'tpr': None,
                'roc_auc': 0,
                'metrics': None,
                'best_params': None
            }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        best_roc_auc = 0
        best_metrics = None
        best_params = None
        best_fpr = None
        best_tpr = None
        best_model = None
        best_optimizer = None

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"\nFold {fold + 1}/{n_splits}")

            try:
                smiles_all, nodes, edges, relations, y = preprocess(dataset_name)
                train_set = GDataset(nodes, edges, relations, y, train_idx, smiles_all)  # 添加smiles_all
                test_set = GDataset(nodes, edges, relations, y, test_idx, smiles_all)    # 添加smiles_all
                
                if len(train_set) == 0 or len(test_set) == 0:
                    print("Empty dataset, skipping fold")
                    continue

                generator = torch.Generator().manual_seed(42)
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
                test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

                for params in ParameterGrid(param_grid):
                    print(f"Testing parameters: {params}")

                    # 分离模型参数和优化器参数
                    model_params = {k: v for k, v in params.items() if k != 'lr'}
                    
                    if model_name == 'RGCN':
                        model = RGCN(num_relations=4, **model_params)  # 关键修复
                    elif model_name == 'GCN':
                        model = GCN(num_relations=1, **model_params)  # 移除了错误的参数传递
                    elif model_name == 'GAT':
                        model = GAT(num_relations=4, **model_params)  # 修正了参数传递方式
                    
                    model.apply(weights_init)
                    model.to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                    
                    train_result = train_single_fold(model, train_loader, test_loader, optimizer, scheduler, EPOCH, weight)
                    accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate(model, test_loader, device)
                    
                    metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'roc_auc': roc_auc
                    }
                    
                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_metrics = metrics
                        best_params = params.copy()
                        best_fpr = fpr
                        best_tpr = tpr
                        best_model = model
                        best_optimizer = optimizer
                        
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
                            print("Warning: Model save failed")
                            
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                continue
                
        return {
            'fpr': best_fpr,
            'tpr': best_tpr,
            'roc_auc': best_roc_auc,
            'metrics': best_metrics,
            'best_params': best_params
        }
        
    except Exception as e:
        print(f"Error in grid_search for {model_name}/{dataset_name}: {str(e)}")
        return {
            'fpr': None,
            'tpr': None,
            'roc_auc': 0,
            'metrics': None,
            'best_params': None
        }
# 定义超参数网格
param_grid = {
    'in_embd': [8],
    'layer_embd': [32, 64],
    'out_embd': [64, 128],
    'dropout': [0.0, 0.2],
    'lr': [1e-3]
}

# 定义数据集列表
datasets = ['JAK1', 'JAK2', 'JAK3', 'TYK2']

# 定义模型列表、训练轮数和批量大小
models = ['GAT']
epochs = [20]
batch_sizes = [64]

# 使用defaultdict自动处理缺失键
all_roc_data = defaultdict(dict)
all_best_metrics = defaultdict(dict)
all_best_params = defaultdict(dict)

# 运行网格搜索
for dataset_name in datasets:
    for model_name, epoch, batch_size in zip(models, epochs, batch_sizes):
        print(f"\n{'='*50}")
        print(f"Running grid search for {model_name} on {dataset_name}")
        print(f"{'='*50}")
        
        try:
            roc_data = grid_search(
                model_name=model_name,
                dataset_name=dataset_name,
                param_grid=param_grid,
                EPOCH=epoch,
                batch_size=batch_size,
                weight=2,
                n_splits=5
            )
            
            # 只有成功返回结果时才保存
            if roc_data['metrics'] is not None and roc_data['best_params'] is not None:
                all_roc_data[dataset_name][model_name] = roc_data
                all_best_metrics[dataset_name][model_name] = roc_data['metrics']
                all_best_params[dataset_name][model_name] = roc_data['best_params']
                print(f"Successfully processed {model_name}/{dataset_name}")
            else:
                print(f"Skipping {model_name}/{dataset_name} due to invalid results")
                
        except Exception as e:
            print(f"Error processing {model_name}/{dataset_name}: {str(e)}")
            continue

# 保存所有结果到文件
try:
    os.makedirs('./results', exist_ok=True)
    with open('./results/best_results.pkl', 'wb') as f:
        pickle.dump({
            'best_metrics': dict(all_best_metrics),
            'best_params': dict(all_best_params),
            'roc_data': dict(all_roc_data)
        }, f)
    print("Results saved to ./results/best_results.pkl")
except Exception as e:
    print(f"Failed to save results: {str(e)}")

# 定义颜色和线型
colors = cycle(['darkorange', 'blue', 'green', 'red'])
linestyles = cycle(['-', '--', '-.', ':'])

# 遍历每个模型，分别绘制其在不同数据集上的 ROC 曲线
for model_name in models:
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # 随机猜测线
    
    for dataset_name in datasets:
        roc_data = all_roc_data.get(dataset_name, {}).get(model_name, {})
        if not roc_data or roc_data['fpr'] is None or roc_data['tpr'] is None:
            print(f"No ROC data for {model_name} on {dataset_name}")
            continue
            
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        roc_auc = roc_data.get('roc_auc', 0)
        
        plt.plot(fpr, tpr, 
                color=next(colors),
                linestyle=next(linestyles),
                linewidth=2,
                label=f'{dataset_name} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {model_name} Model', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs('./roc_curves', exist_ok=True)
    plt.savefig(f'./roc_curves/{model_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# 打印每个模型在每个数据集上的最佳评估指标
for model_name in models:
    print(f"\n{'='*40}")
    print(f"Best evaluation metrics for {model_name}")
    print(f"{'='*40}")
    
    for dataset_name in datasets:
        metrics = all_best_metrics.get(dataset_name, {}).get(model_name, {})
        params = all_best_params.get(dataset_name, {}).get(model_name, {})
        
        if not metrics:
            print(f"  Dataset {dataset_name}: No valid results")
            continue
            
        print(f"\n  Dataset: {dataset_name}")
        print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"    Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"    Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"    F1-score: {metrics.get('f1', 'N/A'):.4f}")
        print(f"    ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        if isinstance(params, dict):
            print("    Best parameters:")
            for k, v in params.items():
                print(f"      {k}: {v}")
        else:
            print("    Best params: Not available")






# #用来输出训练好的模型的架构
# import torch

# # 加载模型（确保使用与训练时相同的设备）
# model = torch.load(r'D:\CC\PycharmProjects\GoGT\saved_models\TYK2\RGCN_best.pth', map_location='cpu')  # 替换为你的模型路径
# print(model)  # 打印模型结构