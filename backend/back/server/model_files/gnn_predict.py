import json
import time
import logging
import traceback
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, RGCNConv, GATConv, GlobalAttention


# 重定向所有非JSON输出到stderr
sys.stderr = sys.__stderr__
# 配置日志到stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# === 全局设置 ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

class PredictConfig:
    BASE_MODEL_DIR = r"D:\Desktop\backend\back\server\model_files\saved_models"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 模型定义（与训练代码严格一致）===
class GCN(nn.Module):
    def __init__(self, in_embd=8, layer_embd=64, out_embd=64, dropout=0.2):
        super(GCN, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(33, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(3, in_embd),
            nn.Embedding(4, in_embd),
            nn.Embedding(2, in_embd),
            nn.Embedding(3, in_embd)
        ])
        self.GCNConv1 = GCNConv(6 * in_embd, layer_embd)
        self.GCNConv2 = GCNConv(layer_embd, out_embd)
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd),
            nn.BatchNorm1d(out_embd),
            nn.ReLU(),
            nn.Linear(out_embd, 1)
        ))
        self.graph_linear = nn.Linear(out_embd, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch):
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            x_ = embds if i == 0 else torch.cat((x_, embds), 1)
        out = self.activation(self.GCNConv1(x_, edge_index))
        out = self.activation(self.GCNConv2(out, edge_index))
        out = self.pool(out, batch)
        out = self.graph_linear(out)
        return torch.sigmoid(out)

class RGCN(nn.Module):
    def __init__(self, in_embd=8, layer_embd=64, out_embd=64, num_relations=4, dropout=0.2):
        super(RGCN, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(33, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(3, in_embd),
            nn.Embedding(4, in_embd),
            nn.Embedding(2, in_embd),
            nn.Embedding(3, in_embd)
        ])
        self.RGCNConv1 = RGCNConv(6 * in_embd, layer_embd, num_relations)
        self.RGCNConv2 = RGCNConv(layer_embd, out_embd, num_relations)
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd),
            nn.BatchNorm1d(out_embd),
            nn.ReLU(),
            nn.Linear(out_embd, 1)
        ))
        self.graph_linear = nn.Linear(out_embd, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch):
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            x_ = embds if i == 0 else torch.cat((x_, embds), 1)
        edge_type = edge_type.long()
        out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
        out = self.activation(self.RGCNConv2(out, edge_index, edge_type))
        out = self.pool(out, batch)
        out = self.graph_linear(out)
        return torch.sigmoid(out)

class GAT(nn.Module):
    def __init__(self, in_embd=8, layer_embd=64, out_embd=64, dropout=0.2):
        super(GAT, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(33, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(3, in_embd),
            nn.Embedding(4, in_embd),
            nn.Embedding(2, in_embd),
            nn.Embedding(3, in_embd)
        ])
        self.GATConv1 = GATConv(6 * in_embd, layer_embd, heads=2, concat=False, dropout=dropout)
        self.GATConv2 = GATConv(layer_embd, out_embd, heads=2, concat=False, dropout=dropout)
        self.activation = nn.Sigmoid()
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(out_embd, out_embd),
            nn.BatchNorm1d(out_embd),
            nn.ReLU(),
            nn.Linear(out_embd, 1)
        ))
        self.graph_linear = nn.Linear(out_embd, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch):
        # 预测时禁用dropout
        if not self.training:
            self.GATConv1.dropout = 0.0
            self.GATConv2.dropout = 0.0

        for i in range(6):
            embds = self.embedding[i](x[:, i])
            x_ = embds if i == 0 else torch.cat((x_, embds), 1)

        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))
        out = self.pool(out, batch)
        out = self.graph_linear(out)

        # 恢复训练设置（如果后续需要）
        if self.training:
            self.GATConv1.dropout = self.dropout
            self.GATConv2.dropout = self.dropout

        return torch.sigmoid(out)

# === 预处理函数（与训练代码严格一致）===
def gen_smiles2graph(sml):
    atom_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17,
                 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]
    dic = {atom_types[i]: i for i in range(len(atom_types))}

    m = Chem.MolFromSmiles(sml)
    if not m:
        return None, None, None

    # 注意键类型映射必须与训练完全一致
    order_string = {
        Chem.rdchem.BondType.SINGLE: 0,  # 训练使用1-4
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    N = len(m.GetAtoms())
    nodes = np.zeros((N, 6))
    for atom in m.GetAtoms():
        nodes[atom.GetIdx()] = [
            dic.get(atom.GetAtomicNum(), 0),
            min(atom.GetDegree(), 4),
            atom.GetFormalCharge() + 1,  # 与训练代码一致：charge偏移
            atom.GetHybridization() - 1, # SP3=2 -> 1
            1 if atom.GetIsAromatic() else 0,
            atom.GetChiralTag()
        ]

    adj = np.zeros((N, N))
    orders = np.zeros((N, N))
    for bond in m.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[u, v] = adj[v, u] = 1
        orders[u, v] = orders[v, u] = order_string.get(bond.GetBondType(), 0)

    return nodes, adj, orders

# === 预测器类 ===
class GNNPredictor:
    def __init__(self, config):
        self.config = config
        self.targets = ['jak1', 'jak2', 'jak3', 'tyk2']
        self.models = {'gcn': {}, 'rgcn': {}, 'gat': {}}
        self._load_models()

    def _load_models(self):
        for model_type in ['gcn', 'rgcn', 'gat']:
            for target in self.targets:
                model_path = os.path.join(
                    self.config.BASE_MODEL_DIR,
                    target.upper(),
                    f"{model_type.upper()}_best.pth"
                )
                if not os.path.exists(model_path):
                    logger.warning(f"{model_type.upper()}模型不存在: {model_path}")
                    continue

                try:
                    checkpoint = torch.load(model_path, map_location=self.config.DEVICE, weights_only=False)

                    # 从checkpoint中提取模型参数（排除不需要的键如'lr'）
                    model_params = {
                        k: v for k, v in checkpoint['params'].items()
                        if k in ['in_embd', 'layer_embd', 'out_embd', 'dropout', 'num_relations']
                    }

                    # 初始化模型
                    if model_type == 'gcn':
                        model = GCN(**model_params)
                    elif model_type == 'rgcn':
                        model = RGCN(**model_params)
                    else:
                        model = GAT(**model_params)

                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.config.DEVICE)
                    model.eval()

                    self.models[model_type][target] = model
                    logger.info(f"成功加载 {target} 的{model_type.upper()}模型")

                except Exception as e:
                    logger.error(f"加载 {target} 的{model_type.upper()}模型失败: {str(e)}")

    def predict(self, smiles, model_type, target):
        if target not in self.models[model_type]:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": f"未加载 {target} 的{model_type.upper()}模型"
            }

        try:
            nodes, adj, orders = gen_smiles2graph(smiles)
            if nodes is None:
                return {"prediction": "error", "probability": 0.0, "error": "SMILES解析失败"}

            # 转换为tensor（与训练代码完全一致）
            x = torch.tensor(nodes, dtype=torch.long).to(self.config.DEVICE)
            edge_index = torch.tensor(np.stack(np.where(adj)), dtype=torch.long).to(self.config.DEVICE)
            edge_type = torch.tensor(orders[adj > 0], dtype=torch.long).to(self.config.DEVICE)
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.config.DEVICE)

            with torch.no_grad():
                prob = self.models[model_type][target](x, edge_index, edge_type, batch).item()

            return {
                "prediction": "active" if prob >= 0.5 else "inactive",
                "probability": prob,
                "error": None
            }

        except Exception as e:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": str(e)
            }

    def validate_consistency(self, val_file_path, model_type):
        """验证指定模型的预测一致性"""
        try:
            val_data = torch.load(val_file_path)
            logger.info(f"开始验证{model_type.upper()}一致性，共 {len(val_data)} 个样本")

            mismatch_count = 0
            for item in val_data:
                result = self.predict(item['smiles'], model_type, item.get('target', 'jak1'))
                if not np.isclose(result['probability'], item['prediction'], atol=1e-6):
                    logger.warning(
                        f"不一致: SMILES {item['smiles']}\n"
                        f"训练预测: {item['prediction']:.6f}\n"
                        f"当前预测: {result['probability']:.6f}"
                    )
                    mismatch_count += 1

            consistency = (len(val_data) - mismatch_count) / len(val_data)
            logger.info(f"{model_type.upper()}验证完成，一致性: {consistency:.2%}")
            return consistency >= 0.999

        except Exception as e:
            logger.error(f"验证失败: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='GNN模型预测脚本')
    parser.add_argument('--smiles', required=True, help='输入的SMILES字符串')
    parser.add_argument('--validate', nargs=2, metavar=('MODEL_TYPE', 'FILE'),
                       help='验证配置，如 --validate gat val_gat.pt')
    args = parser.parse_args()
    try:
        config = PredictConfig()
        predictor = GNNPredictor(config)

        # 验证模式
        if args.validate:
            model_type, val_file = args.validate
            if model_type.lower() not in ['gcn', 'rgcn', 'gat']:
                raise ValueError("模型类型必须是gcn/rgcn/gat")
            if not predictor.validate_consistency(val_file, model_type.lower()):
                raise RuntimeError(f"{model_type.upper()}一致性验证失败")

        # 执行预测
        results = {}
        for target in predictor.targets:
            results[target] = {
                'gcn': predictor.predict(args.smiles, 'gcn', target),
                'rgcn': predictor.predict(args.smiles, 'rgcn', target),
                'gat': predictor.predict(args.smiles, 'gat', target)
            }

        # 确保只输出JSON到stdout
        json.dump({
            "data": results,
            "status": "success",
            "timestamp": int(time.time() * 1000)
        }, sys.stdout, ensure_ascii=False)
        sys.stdout.flush()

    except Exception as e:
        json.dump({
            "error": str(e),
           "traceback": traceback.format_exc(),
            "timestamp": int(time.time() * 1000)
        }, sys.stdout, ensure_ascii=False)
        sys.exit(1)

if __name__ == "__main__":
    main()