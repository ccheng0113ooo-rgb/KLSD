import os
import sys
import json
import warnings
import numpy as np
import torch
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import time
# 禁用所有警告
warnings.filterwarnings("ignore")
# 禁用tqdm进度条
tqdm.disable = True

class Config:
    SAVE_DIR = r"D:\Desktop\backend\back\server\model_files\nn_results_finalall_6"
    TRANSFORMERS_PATH = os.path.join(SAVE_DIR, "transformers_all.joblib")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 1024
    ACTIVITY_THRESHOLD = 6.0

class ActivityValueConverter:
    def __init__(self):
        self.scaler = RobustScaler()
        self.scaler.center_ = np.array([1.82573589])
        self.scaler.scale_ = np.array([2.64453383])
        self.boxcox_lambda = 0.5677683405845432
        self.train_min_activity = 3.050000
        self.epsilon = 1e-6

    def convert_to_original(self, model_output):
        scaled_output = np.asarray(model_output).reshape(-1, 1)
        boxcox_output = self.scaler.inverse_transform(scaled_output).flatten()
        if self.boxcox_lambda == 0:
            shifted_output = np.exp(boxcox_output)
        else:
            shifted_output = (boxcox_output * self.boxcox_lambda + 1) ** (1 / self.boxcox_lambda)
        return shifted_output + self.train_min_activity - self.epsilon

class EnhancedNN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512), torch.nn.GELU(),
            torch.nn.Linear(512, input_dim), torch.nn.Sigmoid()
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1536), torch.nn.BatchNorm1d(1536), torch.nn.GELU(),
            torch.nn.Dropout(0.4), torch.nn.Linear(1536, 768), torch.nn.BatchNorm1d(768), torch.nn.GELU(),
            torch.nn.Linear(768, 384), torch.nn.BatchNorm1d(384), torch.nn.GELU(),
            torch.nn.Linear(384, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return self.net(x * attn_weights).squeeze(-1)

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")

    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, Config.MORGAN_RADIUS, Config.MORGAN_NBITS)
    descriptors = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.HeavyAtomCount(mol), Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol), Descriptors.RingCount(mol),
        rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    ]
    return np.concatenate([morgan, descriptors])

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No SMILES provided"}))
        sys.exit(1)

    smiles = sys.argv[1]
    try:
        # 初始化模型
        transformers = joblib.load(Config.TRANSFORMERS_PATH)
        model = EnhancedNN(1040).to(Config.DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(Config.SAVE_DIR, 'best_nn_model.pth'),
            map_location=Config.DEVICE
        ))
        model.eval()

        # 处理SMILES
        features = smiles_to_features(smiles)

        # 预测
        X_norm = (np.array([features]) - transformers['X_mean']) / (transformers['X_std'] + 1e-6)
        with torch.no_grad():
            pred_norm = model(torch.FloatTensor(X_norm).to(Config.DEVICE)).cpu().numpy()

        # 转换结果
        pred_original = pred_norm * transformers['y_std'] + transformers['y_mean']
        activity_converter = ActivityValueConverter()
        pred_activity = activity_converter.convert_to_original(pred_original)
        is_active = pred_activity >= Config.ACTIVITY_THRESHOLD

        # 构建响应
        result = {
            "success": True,
            "error": None,
            "results": {
                "overall": {
                    "predicted_activity": [float(pred_activity[0])],
                    "is_active": [bool(is_active[0])],
                    "error": None
                }
            },
            "timestamp": int(time.time() * 1000)
        }

        # 确保只输出JSON
        print(json.dumps(result, separators=(',', ':')))
        sys.stdout.flush()

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": int(time.time() * 1000)
        }
        print(json.dumps(error_result, separators=(',', ':')))
        sys.stdout.flush()

if __name__ == "__main__":
    main()