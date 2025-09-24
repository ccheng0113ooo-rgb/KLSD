import json
import time
import logging
import traceback
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, RGCNConv, GATConv, GlobalAttention
from rdkit import Chem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ========== 模型定义 (与训练代码完全一致) ==========
class GCN_VAE(nn.Module):
    def __init__(self, in_embd=4, layer_embd=64, out_embd=64, num_relations=4, dropout=0.2):
        super(GCN_VAE, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(33, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(3, in_embd),
            nn.Embedding(4, in_embd),
            nn.Embedding(2, in_embd),
            nn.Embedding(3, in_embd)
        ])
        self.GCNConv1 = GCNConv(6 * in_embd, layer_embd)
        self.GCNConv2 = GCNConv(layer_embd, out_embd * 2)
        self.activation = nn.Sigmoid()
        self.d = out_embd
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_embd, out_embd),
                nn.BatchNorm1d(out_embd),
                nn.ReLU(),
                nn.Linear(out_embd, 1)
            )
        )
        self.graph_linear = nn.Linear(out_embd, 1)
        self.dropout = nn.Dropout(dropout)
        self.mu_linear = nn.Linear(out_embd * 2, out_embd)
        self.logvar_linear = nn.Linear(out_embd * 2, out_embd)

    def forward(self, x, edge_index, edge_type, batch, type_='finetune'):
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)

        out = self.activation(self.GCNConv1(x_, edge_index))
        out = self.activation(self.GCNConv2(out, edge_index))

        if type_ == 'pretrain':
            mu = self.mu_linear(out)
            logvar = self.logvar_linear(out)
            return None, None, mu, logvar
        else:
            mu = self.mu_linear(out)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            return torch.sigmoid(out)

class RGCN_VAE(nn.Module):
    def __init__(self, in_embd=4, layer_embd=64, out_embd=64, num_relations=4, dropout=0.2):
        super(RGCN_VAE, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(35, in_embd),
            nn.Embedding(10, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(7, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(5, in_embd)
        ])
        self.RGCNConv1 = RGCNConv(6 * in_embd, layer_embd, num_relations)
        self.RGCNConv2 = RGCNConv(layer_embd, out_embd * 2, num_relations)
        self.activation = nn.Sigmoid()
        self.d = out_embd
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_embd, out_embd),
                nn.BatchNorm1d(out_embd),
                nn.ReLU(),
                nn.Linear(out_embd, 1)
            )
        )
        self.graph_linear = nn.Linear(out_embd, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch, type_='finetune'):
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)

        edge_type = edge_type.long()
        out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
        out = self.activation(self.RGCNConv2(out, edge_index, edge_type))

        if type_ == 'pretrain':
            mu = out[:, :self.d]
            logvar = out[:, self.d:]
            return None, None, mu, logvar
        else:
            mu = out[:, :self.d]
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            return torch.sigmoid(out)

class GAT_VAE(nn.Module):
    def __init__(self, in_embd=4, layer_embd=64, out_embd=64, num_relations=4, dropout=0.2):
        super(GAT_VAE, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(35, in_embd),
            nn.Embedding(10, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(7, in_embd),
            nn.Embedding(5, in_embd),
            nn.Embedding(5, in_embd)
        ])
        self.GATConv1 = GATConv(6 * in_embd, layer_embd, heads=4, dropout=dropout)
        self.GATConv2 = GATConv(layer_embd * 4, out_embd * 2, heads=4, dropout=dropout)
        self.activation = nn.Sigmoid()
        self.d = out_embd
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_embd, out_embd),
                nn.BatchNorm1d(out_embd),
                nn.ReLU(),
                nn.Linear(out_embd, 1)
            )
        )
        self.graph_linear = nn.Linear(out_embd, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch, type_='finetune'):
        for i in range(6):
            embds = self.embedding[i](x[:, i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)

        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))

        if type_ == 'pretrain':
            mu = out[:, :self.d]
            logvar = out[:, self.d:]
            return None, None, mu, logvar
        else:
            mu = out[:, :self.d]
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            return torch.sigmoid(out)

# ========== 配置类 ==========
class PredictConfig:
    BASE_MODEL_DIR = r"D:\Desktop\backend\back\server\model_files\saved_models"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_TYPES = ['GCN_VAE', 'RGCN_VAE', 'GAT_VAE']  # 添加支持的模型类型

# ========== 特征工程 (与训练代码完全一致) ==========
def gen_smiles2graph(sml):
    atom_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17,
                 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]
    dic = {atom_types[i]: i for i in range(len(atom_types))}

    m = Chem.MolFromSmiles(sml)
    if not m:
        logger.error(f"无法解析SMILES: {sml}")
        return None, None, None

    order_string = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, 6))

    for atom in m.GetAtoms():
        atom_type = dic.get(atom.GetAtomicNum(), 0)
        degree = min(atom.GetDegree(), 4)
        charge = atom.GetFormalCharge() + 1  # 与训练代码一致
        hybridization = atom.GetHybridization().real - 1  # 与训练代码一致
        is_aromatic = 1 if atom.GetIsAromatic() else 0
        chiral = atom.GetChiralTag()

        nodes[atom.GetIdx()] = [
            atom_type, degree, charge, hybridization, is_aromatic, chiral
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

# ========== 预测器基类 ==========
class BasePredictor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        os.makedirs(self.config.BASE_MODEL_DIR, exist_ok=True)
    def _load_single_model(self, target, model_class, model_name):
        """加载单个模型的通用逻辑"""
        model_path = os.path.join(self.config.BASE_MODEL_DIR, target, f'{model_name}_best.pth')
        config_path = os.path.join(self.config.BASE_MODEL_DIR, target, '{model_name}_preprocess_config.pkl')

        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return None

        try:
            # 加载模型检查点
            checkpoint = torch.load(model_path, map_location=self.config.DEVICE, weights_only=False)

            # 创建模型实例
            model = model_class(**checkpoint['params'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.config.DEVICE)
            model.eval()


            # 加载预处理配置
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    preprocess_config = pickle.load(f)
            else:
                logger.warning(f"预处理配置文件不存在: {config_path}")
                preprocess_config = None

            return {
                'model': model,
                'config': preprocess_config
            }
        except Exception as e:
            logger.error(f"加载 {target} {model_name} 模型失败: {str(e)}")
            return None

    def validate_consistency(self, target, model_name):
        """验证模型一致性的通用逻辑"""
        if target not in self.models or model_name not in self.models[target]:
            logger.warning(f"模型未加载: {target} {model_name}")
            return False

        try:
            model_info = self.models[target][model_name]
            model = model_info['model']

            # 使用测试SMILES验证
            test_smiles = "C1=CC=CC=C1"  # 苯环作为测试用例
            nodes, adj, orders = gen_smiles2graph(test_smiles)
            if nodes is None:
                logger.warning("测试SMILES生成图结构失败")
                return False

            # 修正节点特征索引 (与训练代码一致)
            embedding_sizes = [33, 5, 3, 4, 2, 3] if model_name == 'GCN_VAE' else [35, 10, 5, 7, 5, 5]
            for i in range(6):
                nodes[:, i] = np.clip(nodes[:, i], 0, embedding_sizes[i] - 1)

            # 准备输入数据
            x = torch.tensor(nodes, dtype=torch.long).to(self.config.DEVICE)
            edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long).to(self.config.DEVICE)
            edge_type = torch.tensor(orders[np.nonzero(adj)], dtype=torch.long).to(self.config.DEVICE)
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.config.DEVICE)

            # 两次预测比较
            with torch.no_grad():
                pred1 = model(x, edge_index, edge_type, batch)
                pred2 = model(x, edge_index, edge_type, batch)

            diff = torch.abs(pred1 - pred2).max().item()
            logger.info(f"{model_name} 一致性验证结果 - 最大差异: {diff}")

            return diff < 1e-6
        except Exception as e:
            logger.error(f"{model_name} 一致性验证失败: {str(e)}")
            return False

    def _predict_single(self, smiles, target, model_name):
        """单个模型预测的通用逻辑"""
        if target not in self.models or model_name not in self.models[target]:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": f"未找到 {target} {model_name} 模型"
            }

        # 验证模型一致性
        if not self.validate_consistency(target, model_name):
            logger.warning(f"{target} {model_name} 模型一致性验证未通过")

        try:
            # 生成图结构 (与训练代码一致)
            nodes, adj, orders = gen_smiles2graph(smiles)
            if nodes is None:
                return {
                    "prediction": "error",
                    "probability": 0.0,
                    "error": "无法从SMILES生成图结构"
                }

            # 修正节点特征索引 (与训练代码一致)
            embedding_sizes = [33, 5, 3, 4, 2, 3] if model_name == 'GCN_VAE' else [35, 10, 5, 7, 5, 5]
            for i in range(6):
                nodes[:, i] = np.clip(nodes[:, i], 0, embedding_sizes[i] - 1)

            # 准备输入数据
            x = torch.tensor(nodes, dtype=torch.long).to(self.config.DEVICE)
            edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long).to(self.config.DEVICE)
            edge_type = torch.tensor(orders[np.nonzero(adj)], dtype=torch.long).to(self.config.DEVICE)
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.config.DEVICE)

            # 预测
            with torch.no_grad():
                output = self.models[target][model_name]['model'](x, edge_index, edge_type, batch)
                probability = output.item()
                prediction = "active" if probability >= 0.5 else "inactive"

            return {
                "prediction": prediction,
                "probability": probability,
                "error": None
            }
        except Exception as e:
            logger.error(f"预测过程中发生错误: {str(e)}")
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": str(e)
            }

# ========== GCN预测器 ==========
class GCNPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self._load_models()

    def _load_models(self):
        targets = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
        for target in targets:
            model_info = self._load_single_model(target, GCN_VAE, 'GCN_VAE')
            if model_info:
                if target not in self.models:
                    self.models[target] = {}
                self.models[target]['GCN_VAE'] = model_info
                logger.info(f"成功加载 {target} GCN模型")

    def predict(self, smiles, target):
        return self._predict_single(smiles, target, 'GCN_VAE')

    def predict_all_targets(self, smiles):
        results = {}
        for target in self.models.keys():
            results[target] = {'GCN_VAE': self.predict(smiles, target)}
        return results

# ========== RGCN预测器 ==========
class RGCNPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self._load_models()

    def _load_models(self):
        targets = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
        for target in targets:
            model_info = self._load_single_model(target, RGCN_VAE, 'RGCN_VAE')
            if model_info:
                if target not in self.models:
                    self.models[target] = {}
                self.models[target]['RGCN_VAE'] = model_info
                logger.info(f"成功加载 {target} RGCN模型")

    def predict(self, smiles, target):
        return self._predict_single(smiles, target, 'RGCN_VAE')

    def predict_all_targets(self, smiles):
        results = {}
        for target in self.models.keys():
            results[target] = {'RGCN_VAE': self.predict(smiles, target)}
        return results

# ========== GAT预测器 ==========
class GATPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self._load_models()

    def _load_models(self):
        targets = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
        for target in targets:
            model_info = self._load_single_model(target, GAT_VAE, 'GAT_VAE')
            if model_info:
                if target not in self.models:
                    self.models[target] = {}
                self.models[target]['GAT_VAE'] = model_info
                logger.info(f"成功加载 {target} GAT模型")

    def predict(self, smiles, target):
        return self._predict_single(smiles, target, 'GAT_VAE')

    def predict_all_targets(self, smiles):
        results = {}
        for target in self.models.keys():
            results[target] = {'GAT_VAE': self.predict(smiles, target)}
        return results

# ========== 组合预测器 ==========
class CombinedPredictor:
    def __init__(self, config):
        self.config = config
        self.gcn_predictor = GCNPredictor(config)
        self.rgcn_predictor = RGCNPredictor(config)
        self.gat_predictor = GATPredictor(config)

    def predict_all(self, smiles):
        """使用所有模型进行预测"""
        results = {}

        # 收集各模型的预测结果
        predictors = {
            'GCN': self.gcn_predictor,
            'RGCN': self.rgcn_predictor,
            'GAT': self.gat_predictor
        }

        for name, predictor in predictors.items():
            try:
                pred_results = predictor.predict_all_targets(smiles)
                for target, pred in pred_results.items():
                    if target not in results:
                        results[target] = {}
                    results[target].update(pred)
            except Exception as e:
                logger.error(f"{name} 预测器执行失败: {str(e)}")

        return results

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='GNN模型预测脚本')
    parser.add_argument('--smiles', required=True, help='输入的SMILES字符串')
    args = parser.parse_args()

    try:
        logger.info(f"开始处理SMILES: {args.smiles}")

        # 初始化配置和组合预测器
        config = PredictConfig()
        predictor = CombinedPredictor(config)

        # 执行所有模型对所有靶点的预测
        results = predictor.predict_all(args.smiles)

        response = {
            "data": results,
            "status": "success",
            "timestamp": int(time.time() * 1000)
        }

        print(json.dumps(response, ensure_ascii=False, indent=2))
        sys.stdout.flush()

    except Exception as e:
        error_msg = {
            "error": str(e),
            "type": "fatal_error",
            "traceback": traceback.format_exc(),
            "timestamp": int(time.time() * 1000)
        }
        print(json.dumps(error_msg, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()