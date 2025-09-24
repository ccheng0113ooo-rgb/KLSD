import json
import time
import pickle
import torch
import torch.nn as nn
import logging
import traceback
import os
import sys
import argparse
import base64
import io
from PIL import Image
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, ClassifierMixin
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)
# ========== 必须在全局定义 CNNWrapper ==========
class CNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc_input_dim = self._calculate_fc_input_dim(input_size, hidden_size)
        self.fc = nn.Linear(self.fc_input_dim, output_size)

    def _calculate_fc_input_dim(self, input_size, hidden_size):
        x = torch.randn(1, input_size, 1024)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=1e-5, batch_size=16, weight_decay=1e-2, epochs=20,
                 device='cpu', input_size=1, hidden_size=16, output_size=2,
                 pretrained_model_path=None):
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = torch.device(device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained_model_path = pretrained_model_path

        self.model = CNNClassifier(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_function = torch.nn.CrossEntropyLoss()

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self._load_pretrained()

    def _load_pretrained(self):
        state_dict = torch.load(self.pretrained_model_path, map_location=self.device)
        model_dict = self.model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        if isinstance(X, str):
            X = [X]
        return np.array([self._predict_single(smi) for smi in X])

    def _predict_single(self, smiles):
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is None:
            return 0
        x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)
        with torch.no_grad():
            output = self.model(x)
            _, pred = torch.max(output, 1)
            return pred.item()

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, str):
            X = [X]
        return np.array([self._predict_proba_single(smi) for smi in X])

    def _predict_proba_single(self, smiles):
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is None:
            return 0.0
        x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)
        with torch.no_grad():
            output = self.model(x)
            prob = torch.softmax(output, 1)[0, 1].item()
            return prob
def main():
    parser = argparse.ArgumentParser(description='CNN模型预测脚本')
    parser.add_argument('--smiles', required=True, help='输入的SMILES字符串')
    parser.add_argument('--image_url', help='已有的分子图像URL（Base64编码）')
    args = parser.parse_args()

    try:
        logger.info(f"开始处理SMILES: {args.smiles}")

        # 初始化配置和预测器
        config = PredictConfig()
        predictor = CNNPredictor(config)

        # 检查是否复用图像
        if args.image_url:
            logger.info("检测到已有分子图像，将复用该图像")
            try:
                # 从Base64 URL中提取图像数据
                img_data = base64.b64decode(args.image_url.split(",")[1])
                image = Image.open(io.BytesIO(img_data))
                logger.info("成功加载提供的分子图像")
            except Exception as e:
                logger.error(f"解析提供的图像失败: {str(e)}")
                raise ValueError("提供的图像格式无效")

        # 执行预测
        results = predictor.predict_all_targets(args.smiles)

        # 标记是否复用图像
        results["used_provided_image"] = bool(args.image_url)

        response = {
            "data": results,
            "status": "success",
            "timestamp": int(time.time() * 1000)
        }

        print(json.dumps(response, ensure_ascii=False))
        sys.stdout.flush()

    except Exception as e:
        error_msg = {
            "error": str(e),
            "type": "fatal_error",
            "traceback": traceback.format_exc(),
            "timestamp": int(time.time() * 1000),
            "used_provided_image": False
        }
        print(json.dumps(error_msg, ensure_ascii=False))
        sys.exit(1)

# ========== CNN 模型定义 ==========
class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc_input_dim = self._calculate_fc_input_dim(input_size, hidden_size)
        self.fc = nn.Linear(self.fc_input_dim, output_size)

    def _calculate_fc_input_dim(self, input_size, hidden_size):
        x = torch.randn(1, input_size, 1024)  # 假设输入长度为 1024
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ========== 配置类 ==========
class PredictConfig:
    # 新模型路径
    BASE_MODEL_DIR = r"D:\Desktop\backend\back\server\model_files\cnn_models"
    DEVICE = torch.device("cpu")  # 强制使用CPU以匹配训练环境
    TARGETS = ['jak1', 'jak2', 'jak3', 'tyk2']  # 预测目标

# ========== 特征工程 ==========
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    """将SMILES字符串转换为分子指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error(f"无法解析SMILES: {smiles}")
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fingerprint, dtype=np.uint8)

# ========== CNN预测器 ==========
class CNNPredictor:
    def __init__(self, config):
        self.config = config
        self.targets = config.TARGETS
        self.models = {}  # 存储不同目标的CNN模型
        self._load_models()

    def _load_models(self):
        """加载预训练的CNN模型，适配新路径格式"""
        for target in self.targets:
            # 构建模型路径
            model_filename = f"best_cnn_{target.upper()}.pkl"
            model_path = os.path.join(self.config.BASE_MODEL_DIR, model_filename)

            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                continue

            try:
                # 加载模型
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                # 确保模型在CPU上
                model.device = self.config.DEVICE
                # 确保模型在正确设备上并设置为评估模式
                model.model.to(self.config.DEVICE)
                model.model.eval()
                self.models[target] = model
                logger.info(f"成功加载 {target} 的CNN模型: {model_path}")
            except Exception as e:
                logger.error(f"加载 {target} 的CNN模型失败: {str(e)}")
                self.models[target] = None

    def predict(self, smiles, target):
        """对单个目标进行预测"""
        model = self.models.get(target)
        if model is None:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": f"未加载 {target} 的CNN模型，请检查路径: {os.path.join(self.config.BASE_MODEL_DIR, f'best_cnn_{target.upper()}.pkl')}"
            }

        try:
            # 生成分子指纹
            fingerprint = smiles_to_fingerprint(smiles)
            if fingerprint is None:
                return {
                    "prediction": "error",
                    "probability": 0.0,
                    "error": "无法从SMILES生成分子指纹"
                }

            # 转换为模型输入格式
            x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.config.DEVICE)

            with torch.no_grad():
                output = model.model(x)
                prob = torch.softmax(output, 1)[0, 1].item()
                prediction = "active" if prob >= 0.5 else "inactive"

            return {
                "prediction": prediction,
                "probability": prob,
                "error": None
            }
        except Exception as e:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": f"预测过程出错: {str(e)}"
            }

    def predict_all_targets(self, smiles):
        """对所有目标进行预测"""
        results = {
            "cnn_predictions": {}
        }

        for target in self.targets:
            results["cnn_predictions"][target] = self.predict(smiles, target)

        return results

if __name__ == "__main__":
    main()