

# transformer.py 修改后的完整代码
##(1) 新增图卷积模块(2) 特征融合与互信息计算(融合序列特征和图特征\监控Transformer特征与RDKit特征的冗余性)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mutual_info_score
import numpy as np
import os
import time

class MolecularGraphEncoder(nn.Module):
    """分子图编码器（与序列长度无关）"""
    def __init__(self, atom_dim=64, bond_dim=32):
        super().__init__()
        self.atom_embedding = nn.Embedding(100, atom_dim)
        self.bond_embedding = nn.Embedding(10, bond_dim)
        self.conv1 = GCNConv(atom_dim, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 512)
        
    def smiles_to_graph(self, smi):
        # 实现SMILES到图的转换（需补充具体实现）
        pass
        
    def forward(self, smiles_list):
        batch_features = []
        for smi in smiles_list:
            graph_data = self.smiles_to_graph(smi)
            x = self.atom_embedding(graph_data.x)
            edge_attr = self.bond_embedding(graph_data.edge_attr)
            
            x = F.relu(self.conv1(x, graph_data.edge_index, edge_attr))
            x = F.relu(self.conv2(x, graph_data.edge_index, edge_attr))
            x = self.conv3(x, graph_data.edge_index, edge_attr)
            batch_features.append(x.mean(dim=0))  # 全局平均池化
        return torch.stack(batch_features)

class XFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 必须从外部传入的动态长度
        self.seq_length = kwargs["seq_length"]  
        self.d_model = kwargs.get("d_model", 256)
        self.nhead = kwargs.get("nhead", 8)
        self.num_layers = kwargs.get("num_layers", 6)
        self.dropout = kwargs.get("dropout", 0.1)
        self.use_graph = kwargs.get("use_graph", True)
        self.compute_mi = kwargs.get("compute_mi", True)
        
        # 动态位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_length, self.d_model))
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=2048,
            dropout=self.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 图卷积分支
        if self.use_graph:
            self.graph_encoder = MolecularGraphEncoder()
            self.graph_proj = nn.Linear(512, self.d_model)
            
        # RDKit特征处理
        if self.compute_mi:
            self.rdkit_proj = nn.Linear(200, self.d_model)
            
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

    def forward(self, x, smiles_list=None, rdkit_features=None):
        # 输入校验
        assert x.dim() == 3, f"输入应为[batch, seq, feat]，实际得到{x.shape}"
        assert x.size(1) <= self.seq_length, \
            f"输入长度{x.size(1)}超过模型最大长度{self.seq_length}"
            
        # 动态填充
        if x.size(1) < self.seq_length:
            pad = torch.zeros(x.size(0), self.seq_length-x.size(1), x.size(2))
            x = torch.cat([x, pad.to(x.device)], dim=1)
        
        # 位置编码
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer编码
        seq_features = self.encoder(x)
        
        # 图特征融合
        if self.use_graph and smiles_list:
            graph_features = self.graph_proj(self.graph_encoder(smiles_list))
            seq_features = seq_features + graph_features.unsqueeze(1)
            
        # 互信息计算
        if self.training and self.compute_mi and rdkit_features is not None:
            rdkit_proj = self.rdkit_proj(rdkit_features)
            mi = self._calc_mutual_info(seq_features, rdkit_proj)
            self._log_mi(mi)
            
        return seq_features

    def _calc_mutual_info(self, x, y):
        x_flat = x.detach().cpu().numpy().flatten()
        y_flat = y.detach().cpu().numpy().flatten()
        bins = min(50, int(np.sqrt(len(x_flat))))
        return mutual_info_score(None, None, contingency=np.histogram2d(x_flat, y_flat, bins)[0])

    def _log_mi(self, mi):
        with open("mi_log.txt", "a") as f:
            f.write(f"{time.time()},{mi}\n")

    def get_embeddings(self, x):
        """获取[CLS]位置的分子表示"""
        return self.forward(x)[:, 0, :]