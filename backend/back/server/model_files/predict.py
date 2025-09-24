import os
import sys
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges, ChemicalFeatures
from rdkit import RDConfig  # 新增：导入RDConfig
import warnings
import traceback
from collections import defaultdict
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ========== 1. 配置类 ==========
class PredictConfig:
    SAVE_DIR = r"D:\Desktop\backend\back\server\model_files\optimized_jak_results_finaldata_roc"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MORGAN_RADIUS = 3
    MORGAN_NBITS = 2048
    ATOM_PAIR_BITS = 2048
    TOPOLOGICAL_BITS = 2048
    ACTIVITY_THRESHOLD = 6.0  # 活性阈值

# ========== 2. 模型架构 (与训练代码一致) ==========
class TargetSpecificBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(0.4)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = torch.nn.functional.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x + residual)
        return x

class TargetSpecificModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            TargetSpecificBlock(1024, 1024),
            TargetSpecificBlock(1024, 1024)
        )

        self.jak1_head = self._build_head(1024)
        self.jak2_head = self._build_head(1024)
        self.jak3_head = self._build_head(1024)
        self.tyk2_head = self._build_head(1024, depth=3)

    def _build_head(self, in_dim, depth=2):
        layers = []
        dim = in_dim
        for _ in range(depth-1):
            layers.append(TargetSpecificBlock(dim, dim//2))
            dim = dim // 2
        layers.append(nn.Linear(dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x, target_idx):
        x = self.shared_layers(x)

        if target_idx == 0:
            return self.jak1_head(x).squeeze(-1)
        elif target_idx == 1:
            return self.jak2_head(x).squeeze(-1)
        elif target_idx == 2:
            return self.jak3_head(x).squeeze(-1)
        else:
            return self.tyk2_head(x).squeeze(-1)

# ========== 3. 特征工程 (修复RDConfig导入问题) ==========
def get_pharmacophore_features(mol, target):
    try:
        # 构建特征工厂时使用RDConfig
        factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        features = factory.GetFeaturesForMol(mol)
        feature_counts = defaultdict(int)
        for feat in features:
            feature_counts[feat.GetFamily()] += 1

        return [
            feature_counts['Donor'],
            feature_counts['Acceptor'],
            feature_counts['NegIonizable'],
            feature_counts['PosIonizable'],
            feature_counts['ZnBinder'],
            feature_counts['Aromatic'],
            feature_counts['Hydrophobe'],
            feature_counts['LumpedHydrophobe']
        ]
    except Exception as e:
        print(f"[DEBUG] 药效团特征提取错误 ({target}): {str(e)}", file=sys.stderr)
        return [0] * 8

def get_3d_descriptors(mol, target):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        return [
            AllChem.CalcPMI1(mol),
            AllChem.CalcPMI2(mol),
            AllChem.CalcPMI3(mol),
            AllChem.CalcRadiusOfGyration(mol)
        ]
    except Exception as e:
        print(f"[DEBUG] 3D描述符提取错误 ({target}): {str(e)}", file=sys.stderr)
        return [0.0] * 4

def get_2d_descriptors(mol, target):
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcChi0v(mol),
        rdMolDescriptors.CalcChi1v(mol),
        rdMolDescriptors.CalcChi2v(mol),
        rdMolDescriptors.CalcChi3v(mol),
        rdMolDescriptors.CalcChi4v(mol),
        rdMolDescriptors.CalcHallKierAlpha(mol),
        rdMolDescriptors.CalcKappa1(mol),
        rdMolDescriptors.CalcKappa2(mol),
        rdMolDescriptors.CalcKappa3(mol)
    ]

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        charge_features = [
            np.mean(charges),
            np.std(charges),
            np.max(charges),
            np.min(charges)
        ]
        descriptors.extend(charge_features)
    except Exception as e:
        print(f"[DEBUG] 电荷特征提取错误 ({target}): {str(e)}", file=sys.stderr)
        descriptors.extend([0.0]*4)

    return descriptors

def smiles_to_features(smiles, target_idx, target):
    """增强版特征提取函数，包含详细的调试信息"""
    print(f"[DEBUG] 处理SMILES: {smiles} 针对靶点: {target}", file=sys.stderr)

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"[DEBUG] RDKit无法解析SMILES: {smiles}", file=sys.stderr)
        return None

    print(f"[DEBUG] RDKit成功解析SMILES，开始提取特征...", file=sys.stderr)

    try:
        # 提取摩根指纹
        morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, PredictConfig.MORGAN_RADIUS, PredictConfig.MORGAN_NBITS))
        print(f"[DEBUG] 摩根指纹提取完成，维度: {len(morgan)}", file=sys.stderr)

        # 提取原子对指纹
        atom_pair = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=PredictConfig.ATOM_PAIR_BITS))
        print(f"[DEBUG] 原子对指纹提取完成，维度: {len(atom_pair)}", file=sys.stderr)

        # 提取拓扑扭转指纹
        topological = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=PredictConfig.TOPOLOGICAL_BITS))
        print(f"[DEBUG] 拓扑扭转指纹提取完成，维度: {len(topological)}", file=sys.stderr)

        # 提取RDKit指纹
        rdkit_fp = np.array(AllChem.RDKFingerprint(mol, fpSize=2048))
        print(f"[DEBUG] RDKit指纹提取完成，维度: {len(rdkit_fp)}", file=sys.stderr)

        # 提取2D描述符
        descriptors = get_2d_descriptors(mol, target)
        print(f"[DEBUG] 2D描述符提取完成，维度: {len(descriptors)}", file=sys.stderr)

        # 提取药效团特征
        pharmacophore = get_pharmacophore_features(mol, target)
        print(f"[DEBUG] 药效团特征提取完成，维度: {len(pharmacophore)}", file=sys.stderr)

        # 提取3D描述符
        three_d = get_3d_descriptors(mol, target)
        print(f"[DEBUG] 3D描述符提取完成，维度: {len(three_d)}", file=sys.stderr)

        # 目标特征
        target_feat = np.zeros(4)
        if target_idx is not None:
            target_feat[target_idx] = 1
        print(f"[DEBUG] 目标特征提取完成，维度: {len(target_feat)}", file=sys.stderr)

        # 合并所有特征
        features = np.concatenate([
            morgan,
            atom_pair,
            topological,
            rdkit_fp,
            descriptors,
            pharmacophore,
            three_d,
            target_feat
        ])
        print(f"[DEBUG] 特征提取完成，总维度: {len(features)}", file=sys.stderr)

        return np.nan_to_num(features)

    except Exception as e:
        print(f"[DEBUG] 特征提取过程中发生错误: {str(e)}", file=sys.stderr)
        print(f"[DEBUG] 错误堆栈: {traceback.format_exc()}", file=sys.stderr)
        return None

# ========== 4. 预处理类 ==========
class AdvancedDataPreprocessor:
    def __init__(self):
        self.X_mean = None
        self.X_std = None

    def preprocess(self, X, fit=True):
        if fit:
            self.X_mean = np.nanmean(X, axis=0)
            self.X_std = np.nanstd(X, axis=0)
            self.X_std[self.X_std == 0] = 1.0  # 避免除零错误

        X = np.where(np.isnan(X), self.X_mean, X)
        X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
        return X_norm

# ========== 5. 逆变换函数 ==========
def inverse_transform_prediction(predicted_normalized, target, transformers_path_base):
    try:
        # 修正路径逻辑，使用与模型加载时相同的目录结构
        transformers_path = os.path.join(transformers_path_base, f"transformers_{target}.joblib")
        if not os.path.exists(transformers_path):
            # 尝试使用替代路径（如果需要）
            alt_path = os.path.join(os.path.dirname(transformers_path_base), f"transformers_{target}.joblib")
            if os.path.exists(alt_path):
                transformers_path = alt_path
            else:
                print(f"[DEBUG] 未找到预处理参数文件: {transformers_path}", file=sys.stderr)
                return np.full_like(predicted_normalized, PredictConfig.ACTIVITY_THRESHOLD), np.zeros_like(predicted_normalized, dtype=int)

        transformers = joblib.load(transformers_path)
        print(f"[DEBUG] 成功加载预处理参数文件: {transformers_path}", file=sys.stderr)

        scaler = transformers['scaler']
        boxcox_lambda = transformers['boxcox_lambda']
        train_min_activity = transformers['train_min_activity']
        epsilon = transformers['epsilon']

        pred_norm_2d = np.array(predicted_normalized).reshape(-1, 1)
        boxcox_values = scaler.inverse_transform(pred_norm_2d)

        if boxcox_lambda == 0:
            shifted_values = np.exp(boxcox_values)
        else:
            shifted_values = ((boxcox_values * boxcox_lambda) + 1) ** (1 / boxcox_lambda)

        original_activity = shifted_values.flatten() - epsilon + train_min_activity
        original_activity = np.nan_to_num(original_activity, nan=PredictConfig.ACTIVITY_THRESHOLD)

        # 二分类：≥6为active，<6为inactive
        binary_pred = (original_activity >= PredictConfig.ACTIVITY_THRESHOLD).astype(int)
        return original_activity, binary_pred
    except Exception as e:
        print(f"[DEBUG] 逆变换错误 ({target}): {str(e)}", file=sys.stderr)
        print(f"[DEBUG] 错误堆栈: {traceback.format_exc()}", file=sys.stderr)
        return np.full_like(predicted_normalized, PredictConfig.ACTIVITY_THRESHOLD), np.zeros_like(predicted_normalized, dtype=int)

# ========== 6. 主预测器 ==========
class JakMultiTargetPredictor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.preprocessors = {}
        self.targets = ['jak1', 'jak2', 'jak3', 'tyk2']
        # 直接使用config.SAVE_DIR作为transformers_base，因为预处理文件和模型在同一目录
        self.transformers_base = config.SAVE_DIR
        self._load_models()

    def _load_models(self):
        for target in self.targets:
            print(f"[DEBUG] 开始加载{target}模型...", file=sys.stderr)

            # 加载预处理参数 - 直接使用SAVE_DIR路径
            transformers_path = os.path.join(self.config.SAVE_DIR, f"transformers_{target}.joblib")
            print(f"[DEBUG] 查找预处理参数文件: {transformers_path}", file=sys.stderr)

            if os.path.exists(transformers_path):
                try:
                    transformers = joblib.load(transformers_path)
                    print(f"[DEBUG] 成功加载预处理参数文件: {transformers_path}", file=sys.stderr)
                except Exception as e:
                    print(f"[DEBUG] 加载预处理参数文件失败: {str(e)}", file=sys.stderr)
                    transformers = None
            else:
                print(f"[DEBUG] 预处理参数文件不存在: {transformers_path}", file=sys.stderr)
                transformers = None

            # 加载模型
            model_path = os.path.join(self.config.SAVE_DIR, f"{target}_best_model.pth")
            print(f"[DEBUG] 查找模型文件: {model_path}", file=sys.stderr)

            if not os.path.exists(model_path):
                print(f"[DEBUG] 模型文件不存在: {model_path}", file=sys.stderr)
                continue

            try:
                # 获取输入维度
                if transformers and 'X_mean' in transformers:
                    input_dim = len(transformers['X_mean'])
                    print(f"[DEBUG] 从预处理参数获取输入维度: {input_dim}", file=sys.stderr)
                else:
                    input_dim = 8233  # 默认维度
                    print(f"[DEBUG] 使用默认输入维度: {input_dim}", file=sys.stderr)

                # 初始化模型
                model = TargetSpecificModel(input_dim).to(self.config.DEVICE)
                print(f"[DEBUG] 成功初始化{target}模型", file=sys.stderr)

                # 加载模型权重
                state_dict = torch.load(model_path, map_location=self.config.DEVICE)
                print(f"[DEBUG] 成功加载模型权重: {model_path}", file=sys.stderr)

                model.load_state_dict(state_dict, strict=False)
                print(f"[DEBUG] 模型权重加载完成", file=sys.stderr)

                model.eval()
                print(f"[DEBUG] {target}模型加载完成，设置为评估模式", file=sys.stderr)

                self.models[target] = model
                self.transformers = transformers
                print(f"[DEBUG] {target}模型加载成功", file=sys.stderr)

            except Exception as e:
                print(f"[DEBUG] 加载{target}模型时发生错误: {str(e)}", file=sys.stderr)
                print(f"[DEBUG] 错误堆栈: {traceback.format_exc()}", file=sys.stderr)

    def predict_single_target(self, smiles, target):
        try:
            print(f"[DEBUG] 开始预测{target}...", file=sys.stderr)
            # 1. 特征提取
            target_idx = self.targets.index(target)
            features = smiles_to_features(smiles, target_idx, target)
            if features is None:
                print(f"[DEBUG] {target}特征提取失败，返回Invalid SMILES", file=sys.stderr)
                return {"error": "Invalid SMILES", "predicted_activity": None, "is_active": None}

            print(f"[DEBUG] {target}特征提取成功，维度: {len(features)}", file=sys.stderr)

            # 2. 预处理
            preprocessor = AdvancedDataPreprocessor()

            # 检查是否有预处理参数
            transformers_path = os.path.join(self.config.SAVE_DIR, f"transformers_{target}.joblib")
            if not os.path.exists(transformers_path):
                transformers_path = os.path.join(self.transformers_base, f"transformers_{target}.joblib")

            if os.path.exists(transformers_path):
                try:
                    transformers = joblib.load(transformers_path)
                    if 'X_mean' in transformers and 'X_std' in transformers:
                        preprocessor.X_mean = transformers['X_mean']
                        preprocessor.X_std = transformers['X_std']
                        print(f"[DEBUG] {target}使用预处理参数进行标准化", file=sys.stderr)
                except:
                    print(f"[DEBUG] {target}预处理参数加载失败，使用默认标准化", file=sys.stderr)

            features_norm = preprocessor.preprocess(features.reshape(1, -1), fit=False).flatten()
            print(f"[DEBUG] {target}特征标准化完成", file=sys.stderr)

            # 3. 预测
            features_tensor = torch.FloatTensor(features_norm).unsqueeze(0).to(self.config.DEVICE)
            with torch.no_grad():
                pred_norm = self.models[target](features_tensor, target_idx).cpu().numpy()[0]
            print(f"[DEBUG] {target}模型预测完成，标准化值: {pred_norm}", file=sys.stderr)

            # 4. 逆变换
            original_activity, is_active = inverse_transform_prediction(
                [pred_norm], target, self.transformers_base
            )
            print(f"[DEBUG] {target}逆变换完成，原始活性值: {original_activity[0]}, 活性分类: {is_active[0]}", file=sys.stderr)

            return {
                "predicted_activity": float(original_activity[0]),
                "is_active": bool(is_active[0]),
                "error": None
            }

        except Exception as e:
            print(f"[DEBUG] {target}预测过程中发生错误: {str(e)}", file=sys.stderr)
            print(f"[DEBUG] 错误堆栈: {traceback.format_exc()}", file=sys.stderr)
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "predicted_activity": None,
                "is_active": None
            }

    def predict_all_targets(self, smiles):
        print(f"[DEBUG] 开始预测所有靶点...", file=sys.stderr)
        if not Chem.MolFromSmiles(smiles):
            print(f"[DEBUG] RDKit无法解析SMILES: {smiles}", file=sys.stderr)
            return {"error": "Invalid SMILES", "results": {}}

        print(f"[DEBUG] RDKit成功解析SMILES，开始多靶点预测...", file=sys.stderr)
        results = {}
        for target in self.targets:
            if target not in self.models or self.models[target] is None:
                print(f"[DEBUG] {target}模型未加载，跳过预测", file=sys.stderr)
                results[target] = {
                    "predicted_activity": None,
                    "is_active": None,
                    "error": "Model not loaded"
                }
                continue

            pred = self.predict_single_target(smiles, target)
            results[target] = {
                "predicted_activity": pred["predicted_activity"],
                "is_active": pred["is_active"],
                "error": pred["error"]
            }
        return {"results": results, "error": None}

# ========== 7. 主程序 ==========
# ========== 7. 主程序 ==========
if __name__ == "__main__":
    import time  # 新增：导入time模块用于生成时间戳

    try:
        # 1. 参数检查
        if len(sys.argv) < 2:
            print(json.dumps({
                "success": False,
                "error": "请提供SMILES字符串",
                "traceback": None,
                "results": None,
                "timestamp": int(time.time() * 1000)
            }))
            sys.exit(1)

        smiles = sys.argv[1]
        print(f"[DEBUG] 输入SMILES: {smiles}", file=sys.stderr)

        # 2. 初始化预测器
        config = PredictConfig()
        predictor = JakMultiTargetPredictor(config)

        # 3. 执行预测
        results = predictor.predict_all_targets(smiles)

        # 4. 标准化输出
        output = {
            "success": True,
            "error": results.get("error"),  # 直接使用predict_all_targets返回的error
            "results": results.get("results"),
            "timestamp": int(time.time() * 1000),
            "metadata": {
                "python_version": sys.version,
                "rdkit_version": Chem.rdBase.rdkitVersion,
                "pytorch_version": torch.__version__
            }
        }

        # 5. 确保输出是单行JSON（重要！）
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))

    except json.JSONDecodeError as e:
        error_msg = {
            "success": False,
            "error": f"JSON解析失败: {str(e)}",
            "traceback": traceback.format_exc(),
            "results": None,
            "timestamp": int(time.time() * 1000)
        }
        print(json.dumps(error_msg))
        sys.exit(1)

    except Chem.AtomValenceException as e:
        error_msg = {
            "success": False,
            "error": f"无效的SMILES（原子价错误）: {str(e)}",
            "traceback": traceback.format_exc(),
            "results": None,
            "timestamp": int(time.time() * 1000)
        }
        print(json.dumps(error_msg))
        sys.exit(1)

    except Exception as e:
        error_msg = {
            "success": False,
            "error": f"预测失败: {str(e)}",
            "traceback": traceback.format_exc(),
            "results": None,
            "timestamp": int(time.time() * 1000)
        }
        print(json.dumps(error_msg))
        sys.exit(1)