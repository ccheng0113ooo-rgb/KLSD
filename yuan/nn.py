# ##Test R²: 0.9863
# ##Test RMSE: 0.0355
# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
# from rdkit.Chem.rdmolops import AddHs
# import warnings
# warnings.filterwarnings('ignore')

# # 配置参数
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./nn_results11"
#     SEED = 42
#     USE_GPU = torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if USE_GPU else "cpu")
#     MORGAN_RADIUS = 2
#     MORGAN_NBITS = 1024
#     AUGMENT_TIMES = 3

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)
# np.random.seed(Config.SEED)

# # ------------------------- 数据预处理 -------------------------
# class DataPreprocessor:
#     def __init__(self):
#         self.X_mean = None
#         self.X_std = None
#         self.y_mean = None
#         self.y_std = None
    
#     def preprocess(self, X, y, fit=True):
#         if fit:
#             self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
#             self.y_mean, self.y_std = y.mean(), y.std()
#         X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
#         y_norm = (y - self.y_mean) / (self.y_std + 1e-6)
#         return X_norm, y_norm
    
#     def inverse_transform_y(self, y_norm):
#         return y_norm * self.y_std + self.y_mean

# # ------------------------- 特征工程 -------------------------
# def augment_smiles(smiles, n_augment=3):
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         return [smiles]
#     mol = AddHs(mol)
#     augmented = []
#     for _ in range(n_augment):
#         try:
#             new_mol = Chem.Mol(mol)
#             if AllChem.EmbedMolecule(new_mol, randomSeed=np.random.randint(1000)) == -1:
#                 AllChem.EmbedMolecule(new_mol, randomSeed=np.random.randint(1000), 
#                                     maxAttempts=100, useRandomCoords=True)
#             if AllChem.UFFOptimizeMolecule(new_mol) != 0:
#                 pass
#             augmented.append(Chem.MolToSmiles(new_mol, doRandom=True, canonical=False))
#         except:
#             augmented.append(smiles)
#     return list(set(augmented)) if augmented else [smiles]

# def smiles_to_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         return None
#     morgan = AllChem.GetMorganFingerprintAsBitVect(mol, Config.MORGAN_RADIUS, Config.MORGAN_NBITS)
#     descriptors = [
#         Descriptors.MolWt(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.TPSA(mol),
#         Descriptors.NumHAcceptors(mol),
#         Descriptors.NumHDonors(mol),
#         Descriptors.NumRotatableBonds(mol),
#         rdMolDescriptors.CalcNumAmideBonds(mol),
#         rdMolDescriptors.CalcNumHeterocycles(mol),
#         rdMolDescriptors.CalcNumAromaticRings(mol),
#         Descriptors.HeavyAtomCount(mol),
#         Descriptors.FractionCSP3(mol),
#         rdMolDescriptors.CalcNumAliphaticRings(mol),
#         rdMolDescriptors.CalcNumSpiroAtoms(mol),
#         rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
#         Descriptors.RingCount(mol),
#         rdMolDescriptors.CalcNumAtomStereoCenters(mol)
#     ]
#     return np.concatenate([morgan, descriptors])

# def load_and_featurize():
#     data = pd.read_csv(Config.DATA_PATH)
#     features, labels, smiles_list = [], [], []
#     for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing Molecules"):
#         augmented_smiles = augment_smiles(row['canonical_smiles'], Config.AUGMENT_TIMES)
#         for smi in augmented_smiles:
#             feat = smiles_to_features(smi)
#             if feat is not None:
#                 features.append(feat)
#                 labels.append(row['value_transformed'])
#                 smiles_list.append(smi)
#     return np.array(features), np.array(labels), smiles_list

# # ------------------------- 神经网络模型 -------------------------
# class EnhancedNN(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.GELU(),
#             nn.Linear(512, input_dim),
#             nn.Sigmoid()
#         )
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 1536),
#             nn.BatchNorm1d(1536),
#             nn.GELU(),
#             nn.Dropout(0.4),
#             nn.Linear(1536, 768),
#             nn.BatchNorm1d(768),
#             nn.GELU(),
#             nn.Linear(768, 384),
#             nn.BatchNorm1d(384),
#             nn.GELU(),
#             nn.Linear(384, 1)
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0.01)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         attn_weights = self.attention(x)
#         weighted_x = x * attn_weights
#         return self.net(weighted_x).squeeze(-1)

# # ------------------------- 训练与评估 -------------------------
# def train_nn(X_train, y_train, X_val, y_val, smiles_val):
#     model = EnhancedNN(X_train.shape[1]).to(Config.DEVICE)
#     optimizer = torch.optim.AdamW([
#         {'params': model.attention.parameters(), 'lr': 1e-4},
#         {'params': model.net.parameters(), 'lr': 3e-4}
#     ])
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=8, verbose=True)
#     criterion = nn.HuberLoss(delta=1.0)

#     train_loader = DataLoader(
#         TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
#         batch_size=128, shuffle=True, pin_memory=Config.USE_GPU
#     )
    
#     # 记录训练过程
#     train_losses, val_r2_scores = [], []
#     best_r2 = -np.inf

#     for epoch in range(200):
#         model.train()
#         epoch_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
#             optimizer.zero_grad()
#             pred = model(X_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             epoch_loss += loss.item()
#         train_losses.append(epoch_loss / len(train_loader))

#         # 验证集评估
#         model.eval()
#         with torch.no_grad():
#             val_pred = model(torch.FloatTensor(X_val).to(Config.DEVICE)).cpu().numpy()
#             val_r2 = r2_score(y_val, val_pred)
#             val_r2_scores.append(val_r2)
#             scheduler.step(val_r2)

#             if val_r2 > best_r2:
#                 best_r2 = val_r2
#                 best_weights = model.state_dict()
#                 # 保存验证集预测结果
#                 val_results = pd.DataFrame({
#                     'SMILES': smiles_val,
#                     'True_Value': preprocessor.inverse_transform_y(y_val),
#                     'Predicted_Value': preprocessor.inverse_transform_y(val_pred)
#                 })
#                 val_results.to_csv(os.path.join(Config.SAVE_DIR, 'best_val_predictions.csv'), index=False)

#         print(f"Epoch {epoch+1}/{200} | Train Loss: {train_losses[-1]:.4f} | Val R²: {val_r2:.4f}")

#     # 保存最佳模型
#     torch.save(best_weights, os.path.join(Config.SAVE_DIR, 'best_nn_model.pth'))

#     # 绘制训练曲线
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(val_r2_scores, label='Val R²')
#     plt.xlabel('Epoch')
#     plt.ylabel('R² Score')
#     plt.legend()
#     plt.savefig(os.path.join(Config.SAVE_DIR, 'training_curves.png'))
#     plt.close()

#     return model

# # ------------------------- 主流程 -------------------------
# if __name__ == "__main__":
#     # 1. 加载数据
#     print("Loading and featurizing data...")
#     X_raw, y_raw, smiles_all = load_and_featurize()

#     # 2. 数据分割（保留SMILES信息）
#     X_train_raw, X_test_raw, y_train_raw, y_test_raw, smiles_train, smiles_test = train_test_split(
#         X_raw, y_raw, smiles_all, test_size=0.2, random_state=Config.SEED)
#     X_val_raw, X_test_raw, y_val_raw, y_test_raw, smiles_val, smiles_test = train_test_split(
#         X_test_raw, y_test_raw, smiles_test, test_size=0.5, random_state=Config.SEED)

#     # 3. 数据标准化
#     preprocessor = DataPreprocessor()
#     X_train, y_train = preprocessor.preprocess(X_train_raw, y_train_raw, fit=True)
#     X_val, y_val = preprocessor.preprocess(X_val_raw, y_val_raw, fit=False)
#     X_test, y_test = preprocessor.preprocess(X_test_raw, y_test_raw, fit=False)

#     # **新增：保存特征和标签的统计信息**
#     stats = {
#         'X_mean': preprocessor.X_mean,
#         'X_std': preprocessor.X_std,
#         'y_mean': preprocessor.y_mean,
#         'y_std': preprocessor.y_std
#     }
#     np.savez(
#         os.path.join(Config.SAVE_DIR, 'feature_stats.npz'),
#         **stats
#     )
#     print(f"已保存特征统计信息到 {Config.SAVE_DIR}/feature_stats.npz")

#     # 4. 训练模型
#     print("\nTraining Neural Network...")
#     model = train_nn(X_train, y_train, X_val, y_val, smiles_val)

#     # 5. 测试集评估
#     model.eval()
#     with torch.no_grad():
#         test_pred = model(torch.FloatTensor(X_test).to(Config.DEVICE)).cpu().numpy()
#         test_pred_original = preprocessor.inverse_transform_y(test_pred)
#         test_true_original = preprocessor.inverse_transform_y(y_test)

#         test_r2 = r2_score(test_true_original, test_pred_original)
#         test_rmse = np.sqrt(mean_squared_error(test_true_original, test_pred_original))
#         print(f"\nTest R²: {test_r2:.4f}")
#         print(f"Test RMSE: {test_rmse:.4f}")

#         # 保存测试集结果
#         test_results = pd.DataFrame({
#             'SMILES': smiles_test,
#             'True_Value': test_true_original,
#             'Predicted_Value': test_pred_original
#         })
#         test_results.to_csv(os.path.join(Config.SAVE_DIR, 'test_predictions.csv'), index=False)

#         # 绘制预测 vs 真实值
#         plt.figure(figsize=(8, 6))
#         plt.scatter(test_true_original, test_pred_original, alpha=0.6)
#         plt.plot([min(test_true_original), max(test_true_original)], 
#                  [min(test_true_original), max(test_true_original)], 'r--', lw=2)
#         plt.xlabel('True Values')
#         plt.ylabel('Predicted Values')
#         plt.title(f'NN Prediction (R²={test_r2:.3f})')
#         plt.grid(True)
#         plt.savefig(os.path.join(Config.SAVE_DIR, 'test_prediction_plot.png'), dpi=300)
#         plt.close()


# ##===== Final Results =====
# ##       test_r2  test_rmse  best_val_r2  num_samples
# ##jak1  0.779559   0.468587     0.786732         3462
# ##jak2  0.764318   0.488568     0.730499         5061
# ##jak3  0.691653   0.554131     0.728334         3165
# ##tyk2  0.660604   0.597090     0.631289        11160
# import logging
# import os
# import numpy as np
# import pandas as pd
# from scipy import stats
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges, ChemicalFeatures
# from rdkit import RDConfig
# import warnings
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from torch.cuda.amp import GradScaler, autocast
# from collections import defaultdict
# warnings.filterwarnings('ignore')

# # Enhanced Configuration
# class Config:
#     DATA_PATHS = {
#         'jak1': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak1.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak1.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak1.csv'
#         },
#         'jak2': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak2.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak2.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak2.csv'
#         },
#         'jak3': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak3.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak3.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak3.csv'
#         },
#         'tyk2': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_tyk2.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_tyk2.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_tyk2.csv'
#         }
#     }
#     SAVE_DIR = "./optimized_jak_results_finaldata"
#     SEED = 42
#     USE_GPU = torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if USE_GPU else "cpu")
    
#     # Feature Parameters
#     MORGAN_RADIUS = 3
#     MORGAN_NBITS = 2048
#     ATOM_PAIR_BITS = 2048
#     TOPOLOGICAL_BITS = 2048
    
#     # Training Parameters
#     EARLY_STOP_PATIENCE = 30
#     BATCH_SIZE = 256
#     INIT_LR = 5e-4
#     MIN_LR = 1e-6
#     WEIGHT_DECAY = 1e-5
#     N_EPOCHS = 250
#     DROPOUT_RATE = 0.4
#     TYK2_AUGMENT_FACTOR = 5  # Tyk2数据增强倍数

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)
# np.random.seed(Config.SEED)

# # ------------------------- Enhanced Data Processing -------------------------
# class AdvancedDataPreprocessor:
#     def __init__(self):
#         self.X_mean = None
#         self.X_std = None
#         self.y_mean = None
#         self.y_std = None
    
#     def preprocess(self, X, y, fit=True):
#         y = np.asarray(y).flatten()
#         if fit:
#             self.X_mean = np.nanmean(X, axis=0)
#             self.X_std = np.nanstd(X, axis=0)
#             self.y_mean = np.nanmean(y)
#             self.y_std = np.nanstd(y)
            
#             # Replace NaN std with 1 to avoid division by zero
#             self.X_std[self.X_std == 0] = 1.0
        
#         X = np.where(np.isnan(X), self.X_mean, X)
#         y = np.where(np.isnan(y), self.y_mean, y)
        
#         X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
#         y_norm = (y - self.y_mean) / (self.y_std + 1e-6)
        
#         return X_norm, y_norm
    
#     def inverse_transform_y(self, y_norm):
#         y_norm = np.asarray(y_norm).flatten()
#         return y_norm * self.y_std + self.y_mean

# # ------------------------- Advanced Feature Engineering -------------------------
# def get_pharmacophore_features(mol):
#     """获取药效团特征"""
#     factory = ChemicalFeatures.BuildFeatureFactory(
#         os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
#     try:
#         features = factory.GetFeaturesForMol(mol)
#         feature_counts = defaultdict(int)
#         for feat in features:
#             feature_counts[feat.GetFamily()] += 1
        
#         return [
#             feature_counts['Donor'],
#             feature_counts['Acceptor'],
#             feature_counts['NegIonizable'],
#             feature_counts['PosIonizable'],
#             feature_counts['ZnBinder'],
#             feature_counts['Aromatic'],
#             feature_counts['Hydrophobe'],
#             feature_counts['LumpedHydrophobe']
#         ]
#     except:
#         return [0] * 8

# def get_3d_descriptors(mol):
#     """获取3D描述符（需要3D构象）"""
#     try:
#         mol = Chem.AddHs(mol)
#         AllChem.EmbedMolecule(mol)
#         AllChem.MMFFOptimizeMolecule(mol)
        
#         return [
#             AllChem.CalcPMI1(mol),
#             AllChem.CalcPMI2(mol),
#             AllChem.CalcPMI3(mol),
#             AllChem.CalcRadiusOfGyration(mol)
#         ]
#     except:
#         return [0.0] * 4

# def get_2d_descriptors(mol):
#     """获取2D描述符"""
#     descriptors = [
#         Descriptors.MolWt(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.TPSA(mol),
#         rdMolDescriptors.CalcNumRotatableBonds(mol),
#         rdMolDescriptors.CalcNumAromaticRings(mol),
#         Descriptors.NumHDonors(mol),
#         Descriptors.NumHAcceptors(mol),
#         Descriptors.RingCount(mol),
#         Descriptors.FractionCSP3(mol),
#         rdMolDescriptors.CalcNumHeteroatoms(mol),
#         rdMolDescriptors.CalcNumAmideBonds(mol),
#         rdMolDescriptors.CalcNumAliphaticRings(mol),
#         rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
#         rdMolDescriptors.CalcChi0v(mol),
#         rdMolDescriptors.CalcChi1v(mol),
#         rdMolDescriptors.CalcChi2v(mol),
#         rdMolDescriptors.CalcChi3v(mol),
#         rdMolDescriptors.CalcChi4v(mol),
#         rdMolDescriptors.CalcHallKierAlpha(mol),
#         rdMolDescriptors.CalcKappa1(mol),
#         rdMolDescriptors.CalcKappa2(mol),
#         rdMolDescriptors.CalcKappa3(mol)
#     ]
    
#     # 添加电荷特征
#     try:
#         rdPartialCharges.ComputeGasteigerCharges(mol)
#         charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
#         charge_features = [
#             np.mean(charges),
#             np.std(charges),
#             np.max(charges),
#             np.min(charges)
#         ]
#         descriptors.extend(charge_features)
#     except:
#         descriptors.extend([0.0]*4)
    
#     return descriptors

# def smiles_to_features(smiles, target_idx=None):
#     """从SMILES生成完整特征"""
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         return None
    
#     try:
#         # 指纹特征
#         morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, Config.MORGAN_RADIUS, Config.MORGAN_NBITS))
#         atom_pair = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=Config.ATOM_PAIR_BITS))
#         topological = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=Config.TOPOLOGICAL_BITS))
#         rdkit_fp = np.array(AllChem.RDKFingerprint(mol, fpSize=2048))
        
#         # 描述符特征
#         descriptors = get_2d_descriptors(mol)
#         pharmacophore = get_pharmacophore_features(mol)
#         three_d = get_3d_descriptors(mol)
        
#         # 靶点特异性特征
#         target_feat = np.zeros(4)
#         if target_idx is not None:
#             target_feat[target_idx] = 1
        
#         features = np.concatenate([
#             morgan,
#             atom_pair,
#             topological,
#             rdkit_fp,
#             descriptors,
#             pharmacophore,
#             three_d,
#             target_feat
#         ])
        
#         return np.nan_to_num(features)
#     except:
#         return None

# # ------------------------- Target-Specific Model Architecture -------------------------
# class TargetSpecificBlock(nn.Module):
#     """靶点特异性残差块"""
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.linear1 = nn.Linear(in_features, out_features)
#         self.linear2 = nn.Linear(out_features, out_features)
#         self.norm = nn.LayerNorm(out_features)
#         self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
#         if in_features != out_features:
#             self.shortcut = nn.Linear(in_features, out_features)
#         else:
#             self.shortcut = nn.Identity()
    
#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = F.gelu(self.linear1(x))
#         x = self.dropout(x)
#         x = self.linear2(x)
#         x = self.norm(x + residual)
#         return x

# class TargetSpecificModel(nn.Module):
#     """靶点特异性模型"""
#     def __init__(self, input_dim):
#         super().__init__()
        
#         # 共享特征提取层
#         self.shared_layers = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.GELU(),
#             nn.Dropout(Config.DROPOUT_RATE),
#             TargetSpecificBlock(1024, 1024),
#             TargetSpecificBlock(1024, 1024)
#         )
        
#         # 靶点特异性预测头
#         self.jak1_head = self._build_head(1024)
#         self.jak2_head = self._build_head(1024)
#         self.jak3_head = self._build_head(1024)
#         self.tyk2_head = self._build_head(1024, depth=3)  # Tyk2使用更深的网络
        
#     def _build_head(self, in_dim, depth=2):
#         layers = []
#         dim = in_dim
#         for _ in range(depth-1):
#             layers.append(TargetSpecificBlock(dim, dim//2))
#             dim = dim // 2
#         layers.append(nn.Linear(dim, 1))
#         return nn.Sequential(*layers)
    
#     def forward(self, x, target_idx):
#         x = self.shared_layers(x)
        
#         if target_idx == 0:
#             return self.jak1_head(x).squeeze(-1)
#         elif target_idx == 1:
#             return self.jak2_head(x).squeeze(-1)
#         elif target_idx == 2:
#             return self.jak3_head(x).squeeze(-1)
#         else:
#             return self.tyk2_head(x).squeeze(-1)

# # ------------------------- Advanced Training Pipeline -------------------------
# class AdaptiveLoss(nn.Module):
#     """自适应混合损失函数"""
#     def __init__(self):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的混合参数
    
#     def forward(self, pred, target):
#         mse = F.mse_loss(pred, target)
#         huber = F.smooth_l1_loss(pred, target)
#         return torch.sigmoid(self.alpha) * mse + (1-torch.sigmoid(self.alpha)) * huber

# def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
#                n_epochs, save_path, device, early_stop_patience, target_name):
#     best_val_loss = float('inf')
#     best_r2 = -float('inf')
#     epochs_no_improve = 0
#     scaler = GradScaler()
    
#     # Tyk2特殊处理 - 更频繁的验证
#     validate_every = 1 if target_name == 'tyk2' else 2
    
#     for epoch in range(n_epochs):
#         model.train()
#         train_loss = 0
        
#         for batch in train_loader:
#             optimizer.zero_grad()
            
#             x, y = batch
#             x, y = x.to(device), y.to(device).view(-1)
            
#             with autocast():
#                 pred = model(x, target_indices[target_name])
#                 loss = criterion(pred, y)
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             train_loss += loss.item()
        
#         # 验证阶段
#         if (epoch + 1) % validate_every == 0:
#             val_loss, val_r2 = evaluate(model, val_loader, device, target_name)
#             scheduler.step(val_loss)
            
#             # 保存最佳模型
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_r2 = val_r2
#                 torch.save(model.state_dict(), save_path)
#                 epochs_no_improve = 0
#             else:
#                 epochs_no_improve += validate_every
#                 if epochs_no_improve >= early_stop_patience:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break
            
#             print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss/len(train_loader):.4f} "
#                   f"| Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")
    
#     return best_r2

# def evaluate(model, data_loader, device, target_name):
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for x, y in data_loader:
#             x, y = x.to(device), y.to(device)
#             with autocast():
#                 pred = model(x, target_indices[target_name]).view(-1)
#                 y = y.view(-1)
                
#                 # 过滤无效值
#                 mask = ~(torch.isnan(y) | torch.isnan(pred))
#                 if mask.sum() > 0:
#                     all_preds.append(pred[mask].cpu().numpy())
#                     all_targets.append(y[mask].cpu().numpy())
    
#     if len(all_preds) == 0:
#         return float('inf'), 0.0
    
#     all_preds = np.concatenate(all_preds)
#     all_targets = np.concatenate(all_targets)
    
#     loss = F.mse_loss(torch.FloatTensor(all_preds), torch.FloatTensor(all_targets)).item()
#     r2 = r2_score(all_targets, all_preds)
    
#     return loss, r2

# # ------------------------- Data Augmentation for Tyk2 -------------------------
# def augment_tyk2_data(data, augment_factor=3):
#     """对tyk2数据进行增强"""
#     augmented_data = []
#     for _, row in data.iterrows():
#         smiles = row['canonical_smiles']
#         value = row['value_transformed']
        
#         # 保留原始数据
#         augmented_data.append({'canonical_smiles': smiles, 'value_transformed': value})
        
#         # 数据增强
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             for _ in range(augment_factor):
#                 try:
#                     # SMILES随机化
#                     new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
#                     augmented_data.append({
#                         'canonical_smiles': new_smiles, 
#                         'value_transformed': value * (0.95 + 0.1 * np.random.random())  # 添加轻微噪声
#                     })
#                 except:
#                     continue
    
#     return pd.DataFrame(augmented_data)

# # ------------------------- Main Execution -------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     logger.info("Starting advanced JAK inhibition prediction")
    
#     target_indices = {'jak1': 0, 'jak2': 1, 'jak3': 2, 'tyk2': 3}
#     results = {}
    
#     # 先预训练共享层
#     logger.info("Pre-training shared layers...")
#     pretrain_data = []
#     for target in ['jak1', 'jak2', 'jak3']:
#         data = pd.read_csv(Config.DATA_PATHS[target]['train'])
#         pretrain_data.append(data)
#     pretrain_df = pd.concat(pretrain_data)
    
#     # 对tyk2进行数据增强
#     tyk2_train = pd.read_csv(Config.DATA_PATHS['tyk2']['train'])
#     augmented_tyk2 = augment_tyk2_data(tyk2_train, Config.TYK2_AUGMENT_FACTOR)
    
#     # 为每个靶点训练模型
#     for target, idx in target_indices.items():
#         logger.info(f"\n===== Processing {target.upper()} =====")
        
#         # 加载数据
#         if target == 'tyk2':
#             train_data = augmented_tyk2
#         else:
#             train_data = pd.read_csv(Config.DATA_PATHS[target]['train'])
        
#         val_data = pd.read_csv(Config.DATA_PATHS[target]['val'])
#         test_data = pd.read_csv(Config.DATA_PATHS[target]['test'])
        
#         # 特征提取
#         logger.info("Extracting features...")
#         X_train = []
#         y_train = []
#         for smi, val in tqdm(zip(train_data['canonical_smiles'], train_data['value_transformed']), 
#                             total=len(train_data)):
#             features = smiles_to_features(smi, idx)
#             if features is not None and not np.isnan(val):
#                 X_train.append(features)
#                 y_train.append(val)
        
#         X_val = []
#         y_val = []
#         for smi, val in tqdm(zip(val_data['canonical_smiles'], val_data['value_transformed']), 
#                             total=len(val_data)):
#             features = smiles_to_features(smi, idx)
#             if features is not None and not np.isnan(val):
#                 X_val.append(features)
#                 y_val.append(val)
        
#         X_test = []
#         y_test = []
#         for smi, val in tqdm(zip(test_data['canonical_smiles'], test_data['value_transformed']), 
#                             total=len(test_data)):
#             features = smiles_to_features(smi, idx)
#             if features is not None and not np.isnan(val):
#                 X_test.append(features)
#                 y_test.append(val)
        
#         # 转换为numpy数组
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
#         X_val = np.array(X_val)
#         y_val = np.array(y_val)
#         X_test = np.array(X_test)
#         y_test = np.array(y_test)
        
#         # 预处理
#         preprocessor = AdvancedDataPreprocessor()
#         X_train, y_train = preprocessor.preprocess(X_train, y_train, fit=True)
#         X_val, y_val = preprocessor.preprocess(X_val, y_val, fit=False)
#         X_test, y_test = preprocessor.preprocess(X_test, y_test, fit=False)
        
#         # 创建数据集
#         train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
#         val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
#         test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
#         # 创建数据加载器
#         train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2)
#         test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2)
        
#         # 初始化模型
#         if target == 'jak1':
#             model = TargetSpecificModel(X_train.shape[1]).to(Config.DEVICE)
#         else:
#             # 对于其他靶点，加载jak1的共享层权重
#             model = TargetSpecificModel(X_train.shape[1]).to(Config.DEVICE)
#             if os.path.exists(os.path.join(Config.SAVE_DIR, "jak1_best_model.pth")):
#                 jak1_state = torch.load(os.path.join(Config.SAVE_DIR, "jak1_best_model.pth"))
#                 model.shared_layers.load_state_dict({
#                     k.replace('shared_layers.', ''): v 
#                     for k, v in jak1_state.items() 
#                     if 'shared_layers' in k
#                 }, strict=False)
        
#         # 优化器和调度器
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=Config.INIT_LR * (2.0 if target == 'tyk2' else 1.0),  # Tyk2使用更高学习率
#             weight_decay=Config.WEIGHT_DECAY
#         )
        
#         scheduler = CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=15,
#             T_mult=1,
#             eta_min=Config.MIN_LR
#         )
        
#         # 损失函数
#         criterion = AdaptiveLoss().to(Config.DEVICE) if target == 'tyk2' else nn.SmoothL1Loss()
        
#         # 训练
#         save_path = os.path.join(Config.SAVE_DIR, f"{target}_best_model.pth")
#         best_val_r2 = train_model(
#             model, train_loader, val_loader, 
#             optimizer, scheduler, criterion,
#             Config.N_EPOCHS, save_path,
#             Config.DEVICE, Config.EARLY_STOP_PATIENCE,
#             target
#         )
        
#         # 在测试集上评估
#         model.load_state_dict(torch.load(save_path))
#         test_loss, test_r2 = evaluate(model, test_loader, Config.DEVICE, target)
        
#         # 反标准化预测结果
#         model.eval()
#         with torch.no_grad():
#             test_pred = model(torch.FloatTensor(X_test).to(Config.DEVICE), idx).cpu().numpy()
#             test_pred = preprocessor.inverse_transform_y(test_pred)
#             test_true = preprocessor.inverse_transform_y(y_test)
            
#             result_df = pd.DataFrame({
#                 'SMILES': test_data['canonical_smiles'][:len(test_true)],
#                 'True': test_true,
#                 'Predicted': test_pred,
#                 'Error': np.abs(test_true - test_pred)
#             })
#             result_df.to_csv(os.path.join(Config.SAVE_DIR, f"{target}_predictions.csv"), index=False)
        
#         # 保存结果
#         results[target] = {
#             'test_r2': test_r2,
#             'test_rmse': np.sqrt(test_loss),
#             'best_val_r2': best_val_r2,
#             'num_samples': len(train_data)
#         }
    
#     # 保存和显示最终结果
#     results_df = pd.DataFrame.from_dict(results, orient='index')
#     results_df.to_csv(os.path.join(Config.SAVE_DIR, "final_results.csv"))
    
#     print("\n===== Final Results =====")
#     print(results_df[['test_r2', 'test_rmse', 'best_val_r2', 'num_samples']])
    
#     # 绘制结果
#     plt.figure(figsize=(14, 6))
#     plt.subplot(1, 2, 1)
#     bars = plt.bar(results_df.index, results_df['test_r2'])
#     plt.title('Test R² Scores', fontsize=14)
#     plt.ylabel('R²', fontsize=12)
#     plt.ylim(0.5, 0.9)
    
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.3f}',
#                 ha='center', va='bottom')
    
#     plt.subplot(1, 2, 2)
#     bars = plt.bar(results_df.index, results_df['test_rmse'])
#     plt.title('Test RMSE', fontsize=14)
#     plt.ylabel('RMSE', fontsize=12)
#     plt.ylim(0.3, 0.7)
    
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.3f}',
#                 ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(Config.SAVE_DIR, "performance_summary.png"), dpi=300)
#     plt.close()





##加入阈值划分以及数据转换
##"./optimized_jak_results_finaldata11"
##===== Final Results =====
##       test_r2  test_rmse  best_val_r2  num_samples  Accuracy  Precision    Recall        F1   ROC_AUC
##jak1  0.779095   0.147912     0.788592         3462  0.919481   0.960417  0.943705  0.951988  0.961324
##jak2  0.740167   0.159128     0.722811         5061  0.867299   0.908527  0.917058  0.912773  0.924423
##jak3  0.647803   0.186191     0.677968         3165  0.842654   0.874820  0.885007  0.879884  0.909503
##tyk2  0.648218   0.184140     0.647895        11160  0.885668   0.938596  0.865229  0.900421  0.932162
# import logging
# import os
# import numpy as np
# import pandas as pd
# from scipy import stats
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges, ChemicalFeatures
# from rdkit import RDConfig
# import warnings
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from torch.cuda.amp import GradScaler, autocast
# from collections import defaultdict
# import joblib
# warnings.filterwarnings('ignore')

# # Enhanced Configuration
# class Config:
#     DATA_PATHS = {
#         'jak1': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak1.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak1.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak1.csv'
#         },
#         'jak2': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak2.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak2.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak2.csv'
#         },
#         'jak3': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak3.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak3.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak3.csv'
#         },
#         'tyk2': {
#             'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_tyk2.csv',
#             'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_tyk2.csv',
#             'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_tyk2.csv'
#         }
#     }
#     SAVE_DIR = "./optimized_jak_results_finaldata11"
#     SEED = 42
#     USE_GPU = torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if USE_GPU else "cpu")
    
#     # Feature Parameters
#     MORGAN_RADIUS = 3
#     MORGAN_NBITS = 2048
#     ATOM_PAIR_BITS = 2048
#     TOPOLOGICAL_BITS = 2048
    
#     # Training Parameters
#     EARLY_STOP_PATIENCE = 30
#     BATCH_SIZE = 256
#     INIT_LR = 5e-4
#     MIN_LR = 1e-6
#     WEIGHT_DECAY = 1e-5
#     N_EPOCHS = 250
#     DROPOUT_RATE = 0.4
#     TYK2_AUGMENT_FACTOR = 5  # Tyk2数据增强倍数
#     ACTIVITY_THRESHOLD = 6.0  # 活性阈值

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)
# np.random.seed(Config.SEED)

# # ------------------------- Enhanced Data Processing -------------------------
# class AdvancedDataPreprocessor:
#     def __init__(self):
#         self.X_mean = None
#         self.X_std = None
    
#     def preprocess(self, X, fit=True):
#         """仅对特征X进行预处理，不再处理标签y"""
#         if fit:
#             self.X_mean = np.nanmean(X, axis=0)
#             self.X_std = np.nanstd(X, axis=0)
#             self.X_std[self.X_std == 0] = 1.0  # 避免除零错误
        
#         X = np.where(np.isnan(X), self.X_mean, X)
#         X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
#         return X_norm
    
#     def inverse_transform_X(self, X_norm):
#         """特征逆变换（如有需要）"""
#         return X_norm * self.X_std + self.X_mean

# # ------------------------- 逆变换函数（整合四个靶点） -------------------------
# def inverse_transform_prediction(predicted_normalized, target, transformers_path_base):
#     """
#     将模型输出的标准化值转换为原始活性值，并进行二分类
#     参数：
#     - predicted_normalized: 模型输出的标准化值（一维数组）
#     - target: 靶点名称（'jak1', 'jak2', 'jak3', 'tyk2'）
#     - transformers_path_base: 预处理参数文件基础路径
#     返回：
#     - original_activity: 原始活性值数组
#     - binary_pred: 二分类结果（active/inactive）
#     """
#     try:
#         transformers_path = os.path.join(transformers_path_base, f"transformers_{target}.joblib")
#         transformers = joblib.load(transformers_path)
        
#         scaler = transformers['scaler']
#         boxcox_lambda = transformers['boxcox_lambda']
#         train_min_activity = transformers['train_min_activity']
#         epsilon = transformers['epsilon']
        
#         pred_norm_2d = np.array(predicted_normalized).reshape(-1, 1)
#         boxcox_values = scaler.inverse_transform(pred_norm_2d)
        
#         if boxcox_lambda == 0:
#             shifted_values = np.exp(boxcox_values)
#         else:
#             shifted_values = ((boxcox_values * boxcox_lambda) + 1) ** (1 / boxcox_lambda)
        
#         original_activity = shifted_values.flatten() - epsilon + train_min_activity
        
#         # 处理可能的NaN值
#         original_activity = np.nan_to_num(original_activity, nan=Config.ACTIVITY_THRESHOLD)
        
#         # 二分类：≥6为active，<6为inactive
#         binary_pred = (original_activity >= Config.ACTIVITY_THRESHOLD).astype(int)
#         return original_activity, binary_pred
#     except Exception as e:
#         logging.error(f"逆变换错误 ({target}): {str(e)}")
#         return np.full_like(predicted_normalized, Config.ACTIVITY_THRESHOLD), np.zeros_like(predicted_normalized, dtype=int)

# # ------------------------- 二分类评估指标计算 -------------------------
# def calculate_binary_metrics(y_true, y_pred):
#     """计算二分类指标（Accuracy/Precision/Recall/F1/ROC AUC）"""
#     try:
#         y_true = y_true.flatten()
#         y_pred = y_pred.flatten()
        
#         mask = ~(np.isnan(y_true) | np.isnan(y_pred))
#         y_true = y_true[mask]
#         y_pred = y_pred[mask]
        
#         if len(y_true) == 0 or len(np.unique(y_true)) < 2:
#             logging.warning("二分类指标计算：数据不足或只有单一类别")
#             return {k: np.nan for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']}
        
#         y_true_binary = (y_true >= Config.ACTIVITY_THRESHOLD).astype(int)
#         y_pred_continuous = y_pred
#         y_pred_binary = (y_pred_continuous >= Config.ACTIVITY_THRESHOLD).astype(int)
        
#         accuracy = accuracy_score(y_true_binary, y_pred_binary)
#         precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
#         recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
#         f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        
#         if len(np.unique(y_true_binary)) < 2:
#             logging.warning("二分类指标计算：真实标签中只有单一类别")
#             return {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1': f1,
#                 'ROC_AUC': np.nan
#             }
        
#         roc_auc = roc_auc_score(y_true_binary, y_pred_continuous)
        
#         return {
#             'Accuracy': accuracy,
#             'Precision': precision,
#             'Recall': recall,
#             'F1': f1,
#             'ROC_AUC': roc_auc
#         }
#     except Exception as e:
#         logging.error(f"二分类指标计算错误: {str(e)}")
#         return {k: np.nan for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']}

# # ------------------------- Advanced Feature Engineering -------------------------
# def get_pharmacophore_features(mol):
#     factory = ChemicalFeatures.BuildFeatureFactory(
#         os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
#     try:
#         features = factory.GetFeaturesForMol(mol)
#         feature_counts = defaultdict(int)
#         for feat in features:
#             feature_counts[feat.GetFamily()] += 1
        
#         return [
#             feature_counts['Donor'],
#             feature_counts['Acceptor'],
#             feature_counts['NegIonizable'],
#             feature_counts['PosIonizable'],
#             feature_counts['ZnBinder'],
#             feature_counts['Aromatic'],
#             feature_counts['Hydrophobe'],
#             feature_counts['LumpedHydrophobe']
#         ]
#     except:
#         return [0] * 8

# def get_3d_descriptors(mol):
#     try:
#         mol = Chem.AddHs(mol)
#         AllChem.EmbedMolecule(mol)
#         AllChem.MMFFOptimizeMolecule(mol)
        
#         return [
#             AllChem.CalcPMI1(mol),
#             AllChem.CalcPMI2(mol),
#             AllChem.CalcPMI3(mol),
#             AllChem.CalcRadiusOfGyration(mol)
#         ]
#     except:
#         return [0.0] * 4

# def get_2d_descriptors(mol):
#     descriptors = [
#         Descriptors.MolWt(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.TPSA(mol),
#         rdMolDescriptors.CalcNumRotatableBonds(mol),
#         rdMolDescriptors.CalcNumAromaticRings(mol),
#         Descriptors.NumHDonors(mol),
#         Descriptors.NumHAcceptors(mol),
#         Descriptors.RingCount(mol),
#         Descriptors.FractionCSP3(mol),
#         rdMolDescriptors.CalcNumHeteroatoms(mol),
#         rdMolDescriptors.CalcNumAmideBonds(mol),
#         rdMolDescriptors.CalcNumAliphaticRings(mol),
#         rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
#         rdMolDescriptors.CalcChi0v(mol),
#         rdMolDescriptors.CalcChi1v(mol),
#         rdMolDescriptors.CalcChi2v(mol),
#         rdMolDescriptors.CalcChi3v(mol),
#         rdMolDescriptors.CalcChi4v(mol),
#         rdMolDescriptors.CalcHallKierAlpha(mol),
#         rdMolDescriptors.CalcKappa1(mol),
#         rdMolDescriptors.CalcKappa2(mol),
#         rdMolDescriptors.CalcKappa3(mol)
#     ]
    
#     try:
#         rdPartialCharges.ComputeGasteigerCharges(mol)
#         charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
#         charge_features = [
#             np.mean(charges),
#             np.std(charges),
#             np.max(charges),
#             np.min(charges)
#         ]
#         descriptors.extend(charge_features)
#     except:
#         descriptors.extend([0.0]*4)
    
#     return descriptors

# def smiles_to_features(smiles, target_idx=None):
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         return None
    
#     try:
#         morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, Config.MORGAN_RADIUS, Config.MORGAN_NBITS))
#         atom_pair = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=Config.ATOM_PAIR_BITS))
#         topological = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=Config.TOPOLOGICAL_BITS))
#         rdkit_fp = np.array(AllChem.RDKFingerprint(mol, fpSize=2048))
        
#         descriptors = get_2d_descriptors(mol)
#         pharmacophore = get_pharmacophore_features(mol)
#         three_d = get_3d_descriptors(mol)
        
#         target_feat = np.zeros(4)
#         if target_idx is not None:
#             target_feat[target_idx] = 1
        
#         features = np.concatenate([
#             morgan,
#             atom_pair,
#             topological,
#             rdkit_fp,
#             descriptors,
#             pharmacophore,
#             three_d,
#             target_feat
#         ])
        
#         return np.nan_to_num(features)
#     except:
#         return None

# # ------------------------- Target-Specific Model Architecture -------------------------
# class TargetSpecificBlock(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.linear1 = nn.Linear(in_features, out_features)
#         self.linear2 = nn.Linear(out_features, out_features)
#         self.norm = nn.LayerNorm(out_features)
#         self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
#         if in_features != out_features:
#             self.shortcut = nn.Linear(in_features, out_features)
#         else:
#             self.shortcut = nn.Identity()
    
#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = F.gelu(self.linear1(x))
#         x = self.dropout(x)
#         x = self.linear2(x)
#         x = self.norm(x + residual)
#         return x

# class TargetSpecificModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
        
#         self.shared_layers = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.GELU(),
#             nn.Dropout(Config.DROPOUT_RATE),
#             TargetSpecificBlock(1024, 1024),
#             TargetSpecificBlock(1024, 1024)
#         )
        
#         self.jak1_head = self._build_head(1024)
#         self.jak2_head = self._build_head(1024)
#         self.jak3_head = self._build_head(1024)
#         self.tyk2_head = self._build_head(1024, depth=3)
        
#     def _build_head(self, in_dim, depth=2):
#         layers = []
#         dim = in_dim
#         for _ in range(depth-1):
#             layers.append(TargetSpecificBlock(dim, dim//2))
#             dim = dim // 2
#         layers.append(nn.Linear(dim, 1))
#         return nn.Sequential(*layers)
    
#     def forward(self, x, target_idx):
#         x = self.shared_layers(x)
        
#         if target_idx == 0:
#             return self.jak1_head(x).squeeze(-1)
#         elif target_idx == 1:
#             return self.jak2_head(x).squeeze(-1)
#         elif target_idx == 2:
#             return self.jak3_head(x).squeeze(-1)
#         else:
#             return self.tyk2_head(x).squeeze(-1)

# # ------------------------- Data Augmentation for Tyk2 -------------------------
# def augment_tyk2_data(data, augment_factor=3):
#     """对tyk2数据进行增强"""
#     augmented_data = []
#     for _, row in data.iterrows():
#         smiles = row['canonical_smiles']
#         value = row['value_transformed']
        
#         augmented_data.append({'canonical_smiles': smiles, 'value_transformed': value})
        
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             for _ in range(augment_factor):
#                 try:
#                     new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
#                     augmented_data.append({
#                         'canonical_smiles': new_smiles, 
#                         'value_transformed': value * (0.95 + 0.1 * np.random.random())
#                     })
#                 except:
#                     continue
    
#     return pd.DataFrame(augmented_data)

# # ------------------------- Advanced Training Pipeline -------------------------
# class AdaptiveLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.5))
    
#     def forward(self, pred, target):
#         mse = F.mse_loss(pred, target)
#         huber = F.smooth_l1_loss(pred, target)
#         return torch.sigmoid(self.alpha) * mse + (1-torch.sigmoid(self.alpha)) * huber

# def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
#                n_epochs, save_path, device, early_stop_patience, target_name):
#     best_val_loss = float('inf')
#     best_r2 = -float('inf')
#     epochs_no_improve = 0
#     scaler = GradScaler()
    
#     validate_every = 1 if target_name == 'tyk2' else 2
    
#     for epoch in range(n_epochs):
#         model.train()
#         train_loss = 0
        
#         for batch in train_loader:
#             optimizer.zero_grad()
            
#             x, y = batch
#             x, y = x.to(device), y.to(device).view(-1)
            
#             with autocast():
#                 pred = model(x, target_indices[target_name])
#                 loss = criterion(pred, y)
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             train_loss += loss.item()
        
#         if (epoch + 1) % validate_every == 0:
#             val_loss, val_r2 = evaluate(model, val_loader, device, target_name)
#             scheduler.step(val_loss)
            
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_r2 = val_r2
#                 torch.save(model.state_dict(), save_path)
#                 epochs_no_improve = 0
#             else:
#                 epochs_no_improve += validate_every
#                 if epochs_no_improve >= early_stop_patience:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break
            
#             print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss/len(train_loader):.4f} "
#                   f"| Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")
    
#     return best_r2

# def evaluate(model, data_loader, device, target_name):
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for x, y in data_loader:
#             x, y = x.to(device), y.to(device)
#             with autocast():
#                 pred = model(x, target_indices[target_name]).view(-1)
#                 y = y.view(-1)
                
#                 mask = ~(torch.isnan(y) | torch.isnan(pred))
#                 if mask.sum() > 0:
#                     all_preds.append(pred[mask].cpu().numpy())
#                     all_targets.append(y[mask].cpu().numpy())
    
#     if len(all_preds) == 0:
#         return float('inf'), 0.0
    
#     all_preds = np.concatenate(all_preds)
#     all_targets = np.concatenate(all_targets)
    
#     loss = F.mse_loss(torch.FloatTensor(all_preds), torch.FloatTensor(all_targets)).item()
#     r2 = r2_score(all_targets, all_preds)
    
#     return loss, r2

# # ------------------------- Main Execution -------------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     logger.info("Starting advanced JAK inhibition prediction")
    
#     target_indices = {'jak1': 0, 'jak2': 1, 'jak3': 2, 'tyk2': 3}
#     results = {}
    
#     # 先预训练共享层
#     logger.info("Pre-training shared layers...")
#     pretrain_data = []
#     for target in ['jak1', 'jak2', 'jak3']:
#         data = pd.read_csv(Config.DATA_PATHS[target]['train'])
#         pretrain_data.append(data)
#     pretrain_df = pd.concat(pretrain_data)
    
#     # 对tyk2进行数据增强
#     tyk2_train = pd.read_csv(Config.DATA_PATHS['tyk2']['train'])
#     augmented_tyk2 = augment_tyk2_data(tyk2_train, Config.TYK2_AUGMENT_FACTOR)
    
#     # 为每个靶点训练模型
#     for target, idx in target_indices.items():
#         logger.info(f"\n===== Processing {target.upper()} =====")
        
#         if target == 'tyk2':
#             train_data = augmented_tyk2
#         else:
#             train_data = pd.read_csv(Config.DATA_PATHS[target]['train'])
        
#         val_data = pd.read_csv(Config.DATA_PATHS[target]['val'])
#         test_data = pd.read_csv(Config.DATA_PATHS[target]['test'])
        
#         logger.info("Extracting features...")
#         X_train = []
#         y_train = []
#         for smi, val in tqdm(zip(train_data['canonical_smiles'], train_data['value_transformed']), 
#                             total=len(train_data)):
#             features = smiles_to_features(smi, idx)
#             if features is not None and not np.isnan(val):
#                 X_train.append(features)
#                 y_train.append(val)
        
#         X_val = []
#         y_val = []
#         for smi, val in tqdm(zip(val_data['canonical_smiles'], val_data['value_transformed']), 
#                             total=len(val_data)):
#             features = smiles_to_features(smi, idx)
#             if features is not None and not np.isnan(val):
#                 X_val.append(features)
#                 y_val.append(val)
        
#         X_test = []
#         y_test = []
#         for smi, val in tqdm(zip(test_data['canonical_smiles'], test_data['value_transformed']), 
#                             total=len(test_data)):
#             features = smiles_to_features(smi, idx)
#             if features is not None and not np.isnan(val):
#                 X_test.append(features)
#                 y_test.append(val)
        
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
#         X_val = np.array(X_val)
#         y_val = np.array(y_val)
#         X_test = np.array(X_test)
#         y_test = np.array(y_test)
        
#         preprocessor = AdvancedDataPreprocessor()
#         X_train = preprocessor.preprocess(X_train, fit=True)
#         X_val = preprocessor.preprocess(X_val, fit=False)
#         X_test = preprocessor.preprocess(X_test, fit=False)
        
#         train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
#         val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
#         test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
#         train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2)
#         test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2)
        
#         if target == 'jak1':
#             model = TargetSpecificModel(X_train.shape[1]).to(Config.DEVICE)
#         else:
#             model = TargetSpecificModel(X_train.shape[1]).to(Config.DEVICE)
#             if os.path.exists(os.path.join(Config.SAVE_DIR, "jak1_best_model.pth")):
#                 jak1_state = torch.load(os.path.join(Config.SAVE_DIR, "jak1_best_model.pth"))
#                 model.shared_layers.load_state_dict({
#                     k.replace('shared_layers.', ''): v 
#                     for k, v in jak1_state.items() 
#                     if 'shared_layers' in k
#                 }, strict=False)
        
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=Config.INIT_LR * (2.0 if target == 'tyk2' else 1.0),
#             weight_decay=Config.WEIGHT_DECAY
#         )
        
#         scheduler = CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=15,
#             T_mult=1,
#             eta_min=Config.MIN_LR
#         )
        
#         criterion = AdaptiveLoss().to(Config.DEVICE) if target == 'tyk2' else nn.SmoothL1Loss()
        
#         save_path = os.path.join(Config.SAVE_DIR, f"{target}_best_model.pth")
#         best_val_r2 = train_model(
#             model, train_loader, val_loader, 
#             optimizer, scheduler, criterion,
#             Config.N_EPOCHS, save_path,
#             Config.DEVICE, Config.EARLY_STOP_PATIENCE,
#             target
#         )
        
#         model.load_state_dict(torch.load(save_path))
#         test_loss, test_r2 = evaluate(model, test_loader, Config.DEVICE, target)
        
#         # 获取预处理参数路径
#         transformers_path_base = os.path.dirname(Config.DATA_PATHS[target]['train'])
        
#         # 反标准化预测结果
#         model.eval()
#         with torch.no_grad():
#             test_pred_normalized = model(torch.FloatTensor(X_test).to(Config.DEVICE), idx).cpu().numpy()
            
#             # 逆变换为原始活性值
#             test_pred_original, test_pred_binary = inverse_transform_prediction(
#                 test_pred_normalized,
#                 target=target,
#                 transformers_path_base=transformers_path_base
#             )
            
#             # 直接从测试数据获取真实原始活性值
#             test_true_original = test_data['activity'].values[:len(test_pred_original)]
            
#             # 保存结果
#             result_df = pd.DataFrame({
#                 'SMILES': test_data['canonical_smiles'][:len(test_true_original)],
#                 'True_Activity': test_true_original,
#                 'Predicted_Activity': test_pred_original,
#                 'Predicted_Active': test_pred_binary,
#                 'Error': np.abs(test_true_original - test_pred_original)
#             })
#             result_df.to_csv(os.path.join(Config.SAVE_DIR, f"{target}_predictions.csv"), index=False)
        
#         # 计算二分类指标
#         binary_metrics = calculate_binary_metrics(test_true_original, test_pred_original)
        
#         # 保存结果
#         results[target] = {
#             'test_r2': test_r2,
#             'test_rmse': np.sqrt(test_loss),
#             'best_val_r2': best_val_r2,
#             'num_samples': len(train_data),
#             **binary_metrics
#         }
    
#     # 保存和显示最终结果
#     results_df = pd.DataFrame.from_dict(results, orient='index')
#     results_df.to_csv(os.path.join(Config.SAVE_DIR, "final_results.csv"))
    
#     print("\n===== Final Results =====")
#     print(results_df[['test_r2', 'test_rmse', 'best_val_r2', 'num_samples',
#                       'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']])
    
#     # 绘制结果
#     plt.figure(figsize=(18, 8))
    
#     plt.subplot(2, 2, 1)
#     bars = plt.bar(results_df.index, results_df['test_r2'])
#     plt.title('Test R² Scores', fontsize=12)
#     plt.ylabel('R²', fontsize=10)
#     plt.ylim(0.5, 0.9)
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.3f}',
#                 ha='center', va='bottom', fontsize=8)
    
#     plt.subplot(2, 2, 2)
#     bars = plt.bar(results_df.index, results_df['test_rmse'])
#     plt.title('Test RMSE', fontsize=12)
#     plt.ylabel('RMSE', fontsize=10)
#     plt.ylim(0.3, 0.7)
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.3f}',
#                 ha='center', va='bottom', fontsize=8)
    
#     plt.subplot(2, 2, 3)
#     binary_metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1']
#     bars = results_df[binary_metrics_cols].plot(kind='bar', ax=plt.gca())
#     plt.title('Binary Classification Metrics', fontsize=12)
#     plt.ylabel('Score', fontsize=10)
#     plt.ylim(0, 1.0)
#     plt.legend(title='Metrics', bbox_to_anchor=(1, 1))
#     for container in bars.containers:
#         for bar in container:
#             height = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{height:.3f}',
#                     ha='center', va='bottom', fontsize=8)
    
#     plt.subplot(2, 2, 4)
#     bars = plt.bar(results_df.index, results_df['ROC_AUC'])
#     plt.title('ROC AUC', fontsize=12)
#     plt.ylabel('AUC', fontsize=10)
#     plt.ylim(0.5, 1.0)
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.3f}',
#                 ha='center', va='bottom', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(Config.SAVE_DIR, "performance_summary.png"), dpi=300)
#     plt.close()








##加入roc曲线
##===== Final Results =====
##       test_r2  test_rmse  best_val_r2  num_samples  Accuracy  Precision    Recall        F1   ROC_AUC
##jak1  0.779095   0.147912     0.788592         3462  0.919481   0.960417  0.943705  0.951988  0.961324
##jak2  0.740167   0.159128     0.722811         5061  0.867299   0.908527  0.917058  0.912773  0.924423
##jak3  0.647803   0.186191     0.677968         3165  0.842654   0.874820  0.885007  0.879884  0.909503
##tyk2  0.648218   0.184140     0.647895        11160  0.885668   0.938596  0.865229  0.900421  0.932162
import logging
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import auc, r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges, ChemicalFeatures
from rdkit import RDConfig
import warnings
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import joblib
from itertools import cycle
warnings.filterwarnings('ignore')

# Enhanced Configuration
class Config:
    DATA_PATHS = {
        'jak1': {
            'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak1.csv',
            'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak1.csv',
            'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak1.csv'
        },
        'jak2': {
            'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak2.csv',
            'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak2.csv',
            'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak2.csv'
        },
        'jak3': {
            'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_jak3.csv',
            'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_jak3.csv',
            'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_jak3.csv'
        },
        'tyk2': {
            'train': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\train_processed_tyk2.csv',
            'val': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\val_processed_tyk2.csv',
            'test': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed\test_processed_tyk2.csv'
        }
    }
    SAVE_DIR = "./optimized_jak_results_finaldata_roc"
    SEED = 42
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_GPU else "cpu")
    
    # Feature Parameters
    MORGAN_RADIUS = 3
    MORGAN_NBITS = 2048
    ATOM_PAIR_BITS = 2048
    TOPOLOGICAL_BITS = 2048
    
    # Training Parameters
    EARLY_STOP_PATIENCE = 30
    BATCH_SIZE = 256
    INIT_LR = 5e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5
    N_EPOCHS = 250
    DROPOUT_RATE = 0.4
    TYK2_AUGMENT_FACTOR = 5  # Tyk2数据增强倍数
    ACTIVITY_THRESHOLD = 6.0  # 活性阈值

os.makedirs(Config.SAVE_DIR, exist_ok=True)
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ------------------------- Enhanced Data Processing -------------------------
class AdvancedDataPreprocessor:
    def __init__(self):
        self.X_mean = None
        self.X_std = None
    
    def preprocess(self, X, fit=True):
        """仅对特征X进行预处理，不再处理标签y"""
        if fit:
            self.X_mean = np.nanmean(X, axis=0)
            self.X_std = np.nanstd(X, axis=0)
            self.X_std[self.X_std == 0] = 1.0  # 避免除零错误
        
        X = np.where(np.isnan(X), self.X_mean, X)
        X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
        return X_norm
    
    def inverse_transform_X(self, X_norm):
        """特征逆变换（如有需要）"""
        return X_norm * self.X_std + self.X_mean

# ------------------------- 新增函数：更新.joblib文件 -------------------------

def update_transformers_with_stats(joblib_path, x_mean, x_std, y_mean, y_std):
    """
    更新现有的transformers.joblib文件，添加X和y的统计量
    保留原始内容，仅追加新字段
    """
    try:
        # 加载现有transformers
        transformers = joblib.load(joblib_path)
        
        # 添加新字段（如果已存在则跳过）
        updated = False
        if 'X_mean' not in transformers:
            transformers['X_mean'] = x_mean
            transformers['X_std'] = x_std
            updated = True
        if 'y_mean' not in transformers:
            transformers['y_mean'] = y_mean
            transformers['y_std'] = y_std
            updated = True
            
        if updated:
            joblib.dump(transformers, joblib_path)
            print(f"\nUpdated transformers with stats at: {joblib_path}")
        else:
            print(f"\nAll stats already exist in transformers, skip updating.")
            
    except FileNotFoundError:
        print(f"\nWarning: Transformers file not found at {joblib_path}. Skip updating.")
    except Exception as e:
        print(f"\nWarning: Failed to update transformers - {str(e)}")

# ------------------------- 逆变换函数（整合四个靶点） -------------------------
def inverse_transform_prediction(predicted_normalized, target, transformers_path_base):
    """
    将模型输出的标准化值转换为原始活性值，并进行二分类
    参数：
    - predicted_normalized: 模型输出的标准化值（一维数组）
    - target: 靶点名称（'jak1', 'jak2', 'jak3', 'tyk2'）
    - transformers_path_base: 预处理参数文件基础路径
    返回：
    - original_activity: 原始活性值数组
    - binary_pred: 二分类结果（active/inactive）
    """
    try:
        transformers_path = os.path.join(transformers_path_base, f"transformers_{target}.joblib")
        transformers = joblib.load(transformers_path)
        
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
        
        # 处理可能的NaN值
        original_activity = np.nan_to_num(original_activity, nan=Config.ACTIVITY_THRESHOLD)
        
        # 二分类：≥6为active，<6为inactive
        binary_pred = (original_activity >= Config.ACTIVITY_THRESHOLD).astype(int)
        return original_activity, binary_pred
    except Exception as e:
        logging.error(f"逆变换错误 ({target}): {str(e)}")
        return np.full_like(predicted_normalized, Config.ACTIVITY_THRESHOLD), np.zeros_like(predicted_normalized, dtype=int)

# ------------------------- 二分类评估指标计算 -------------------------
def calculate_binary_metrics(y_true, y_pred, threshold=None):
    """
    计算二分类指标（Accuracy/Precision/Recall/F1/ROC AUC）
    
    参数:
        y_true: 真实活性值数组
        y_pred: 预测活性值数组
        threshold: 活性阈值（可选，默认使用Config.ACTIVITY_THRESHOLD）
    
    返回:
        包含各项指标的字典
    """
    try:
        # 确定使用的阈值
        active_threshold = threshold if threshold is not None else Config.ACTIVITY_THRESHOLD
        
        # 数据预处理
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # 过滤无效值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # 检查数据有效性
        if len(y_true) == 0:
            logging.warning("二分类指标计算：无有效数据")
            return {k: np.nan for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']}
        
        # 转换为二分类标签
        y_true_binary = (y_true >= active_threshold).astype(int)
        y_pred_continuous = y_pred  # 保留连续值用于ROC计算
        y_pred_binary = (y_pred_continuous >= active_threshold).astype(int)
        
        # 计算基础指标
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, 
                                  average='binary', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, 
                            average='binary', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, 
                     average='binary', zero_division=0)
        
        # 处理单一类别情况
        if len(np.unique(y_true_binary)) < 2:
            logging.warning("二分类指标计算：真实标签中只有单一类别")
            return {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC_AUC': np.nan,
                'Threshold_Used': active_threshold  # 新增字段，记录使用的阈值
            }
        
        # 计算ROC AUC
        roc_auc = roc_auc_score(y_true_binary, y_pred_continuous)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC_AUC': roc_auc,
            'Threshold_Used': active_threshold  # 新增字段，记录使用的阈值
        }
        
    except Exception as e:
        logging.error(f"二分类指标计算错误: {str(e)}", exc_info=True)
        return {
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1': np.nan,
            'ROC_AUC': np.nan,
            'Threshold_Used': active_threshold if 'active_threshold' in locals() else None
        }
# ------------------------- Advanced Feature Engineering -------------------------
def get_pharmacophore_features(mol):
    factory = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    try:
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
    except:
        return [0] * 8

def get_3d_descriptors(mol):
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
    except:
        return [0.0] * 4

def get_2d_descriptors(mol):
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
    except:
        descriptors.extend([0.0]*4)
    
    return descriptors

def smiles_to_features(smiles, target_idx=None):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    try:
        morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, Config.MORGAN_RADIUS, Config.MORGAN_NBITS))
        atom_pair = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=Config.ATOM_PAIR_BITS))
        topological = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=Config.TOPOLOGICAL_BITS))
        rdkit_fp = np.array(AllChem.RDKFingerprint(mol, fpSize=2048))
        
        descriptors = get_2d_descriptors(mol)
        pharmacophore = get_pharmacophore_features(mol)
        three_d = get_3d_descriptors(mol)
        
        target_feat = np.zeros(4)
        if target_idx is not None:
            target_feat[target_idx] = 1
        
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
        
        return np.nan_to_num(features)
    except:
        return None

# ------------------------- Target-Specific Model Architecture -------------------------
class TargetSpecificBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.linear1(x))
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
            nn.Dropout(Config.DROPOUT_RATE),
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

# ------------------------- Data Augmentation for Tyk2 -------------------------
def augment_tyk2_data(data, augment_factor=3):
    """对tyk2数据进行增强"""
    augmented_data = []
    for _, row in data.iterrows():
        smiles = row['canonical_smiles']
        value = row['value_transformed']
        
        augmented_data.append({'canonical_smiles': smiles, 'value_transformed': value})
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for _ in range(augment_factor):
                try:
                    new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    augmented_data.append({
                        'canonical_smiles': new_smiles, 
                        'value_transformed': value * (0.95 + 0.1 * np.random.random())
                    })
                except:
                    continue
    
    return pd.DataFrame(augmented_data)

# ------------------------- Advanced Training Pipeline -------------------------
class AdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        huber = F.smooth_l1_loss(pred, target)
        return torch.sigmoid(self.alpha) * mse + (1-torch.sigmoid(self.alpha)) * huber

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
               n_epochs, save_path, device, early_stop_patience, target_name):
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    epochs_no_improve = 0
    scaler = GradScaler()
    
    validate_every = 1 if target_name == 'tyk2' else 2
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            x, y = batch
            x, y = x.to(device), y.to(device).view(-1)
            
            with autocast():
                pred = model(x, target_indices[target_name])
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        if (epoch + 1) % validate_every == 0:
            val_loss, val_r2 = evaluate(model, val_loader, device, target_name)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_r2 = val_r2
                torch.save(model.state_dict(), save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += validate_every
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss/len(train_loader):.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")
    
    return best_r2

def evaluate(model, data_loader, device, target_name):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                pred = model(x, target_indices[target_name]).view(-1)
                y = y.view(-1)
                
                mask = ~(torch.isnan(y) | torch.isnan(pred))
                if mask.sum() > 0:
                    all_preds.append(pred[mask].cpu().numpy())
                    all_targets.append(y[mask].cpu().numpy())
    
    if len(all_preds) == 0:
        return float('inf'), 0.0
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    loss = F.mse_loss(torch.FloatTensor(all_preds), torch.FloatTensor(all_targets)).item()
    r2 = r2_score(all_targets, all_preds)
    
    return loss, r2

# ------------------------- ROC曲线绘制函数 -------------------------
def plot_roc_curve(y_true, y_score, target_name, save_path):
    """绘制ROC曲线，样式与之前保持一致"""
    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    
    # 定义颜色和线型循环
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    
    # 绘制当前靶点的ROC曲线
    plt.plot(fpr, tpr, 
             color=next(colors),
             linestyle=next(linestyles),
             linewidth=2,
             label=f'{target_name} (AUC = {roc_auc:.2f})')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for Target Specific Model on {target_name.upper()} Dataset', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    os.makedirs('./roc_curves', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------- Main Execution -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting advanced JAK inhibition prediction")
    
    target_indices = {'jak1': 0, 'jak2': 1, 'jak3': 2, 'tyk2': 3}
    results = {}
    
    # 先预训练共享层
    logger.info("Pre-training shared layers...")
    pretrain_data = []
    for target in ['jak1', 'jak2', 'jak3']:
        data = pd.read_csv(Config.DATA_PATHS[target]['train'])
        pretrain_data.append(data)
    pretrain_df = pd.concat(pretrain_data)
    
    # 对tyk2进行数据增强
    tyk2_train = pd.read_csv(Config.DATA_PATHS['tyk2']['train'])
    augmented_tyk2 = augment_tyk2_data(tyk2_train, Config.TYK2_AUGMENT_FACTOR)
    
    # 为每个靶点训练模型
    for target, idx in target_indices.items():
        logger.info(f"\n===== Processing {target.upper()} =====")
        
        if target == 'tyk2':
            train_data = augmented_tyk2
        else:
            train_data = pd.read_csv(Config.DATA_PATHS[target]['train'])
        
        val_data = pd.read_csv(Config.DATA_PATHS[target]['val'])
        test_data = pd.read_csv(Config.DATA_PATHS[target]['test'])
        
        logger.info("Extracting features...")
        X_train = []
        y_train = []
        for smi, val in tqdm(zip(train_data['canonical_smiles'], train_data['value_transformed']), 
                            total=len(train_data)):
            features = smiles_to_features(smi, idx)
            if features is not None and not np.isnan(val):
                X_train.append(features)
                y_train.append(val)
        
        X_val = []
        y_val = []
        for smi, val in tqdm(zip(val_data['canonical_smiles'], val_data['value_transformed']), 
                            total=len(val_data)):
            features = smiles_to_features(smi, idx)
            if features is not None and not np.isnan(val):
                X_val.append(features)
                y_val.append(val)
        
        X_test = []
        y_test = []
        for smi, val in tqdm(zip(test_data['canonical_smiles'], test_data['value_transformed']), 
                            total=len(test_data)):
            features = smiles_to_features(smi, idx)
            if features is not None and not np.isnan(val):
                X_test.append(features)
                y_test.append(val)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        preprocessor = AdvancedDataPreprocessor()
        X_train = preprocessor.preprocess(X_train, fit=True)  # 这里会计算X_mean和X_std
        
        # 计算y的均值和标准差
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_mean = y_scaler.mean_[0]
        y_std = np.sqrt(y_scaler.var_[0])
        
        X_val = preprocessor.preprocess(X_val, fit=False)
        X_test = preprocessor.preprocess(X_test, fit=False)
        
        # ========== 更新transformers.joblib文件 ==========
        transformers_path = os.path.join(os.path.dirname(Config.DATA_PATHS[target]['train']), 
                                       f"transformers_{target}.joblib")
        if os.path.exists(transformers_path):
            # 加载现有transformers
            transformers = joblib.load(transformers_path)
            
            # 更新统计量
            transformers['X_mean'] = preprocessor.X_mean
            transformers['X_std'] = preprocessor.X_std
            transformers['y_mean'] = y_mean
            transformers['y_std'] = y_std
            
            # 保存更新后的transformers
            joblib.dump(transformers, transformers_path)
            logger.info(f"Updated transformers with X and y stats at: {transformers_path}")
        else:
            logger.warning(f"{transformers_path} not found, creating new transformers file")
            transformers = {
                'X_mean': preprocessor.X_mean,
                'X_std': preprocessor.X_std,
                'y_mean': y_mean,
                'y_std': y_std,
                # 保留其他已有参数
                **({} if not os.path.exists(transformers_path) else joblib.load(transformers_path))
            }
            joblib.dump(transformers, transformers_path)
        # ========== 更新结束 ==========
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(
            y_scaler.transform(y_val.reshape(-1, 1)).flatten()))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(
            y_scaler.transform(y_test.reshape(-1, 1)).flatten()))
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2)
        
        if target == 'jak1':
            model = TargetSpecificModel(X_train.shape[1]).to(Config.DEVICE)
        else:
            model = TargetSpecificModel(X_train.shape[1]).to(Config.DEVICE)
            if os.path.exists(os.path.join(Config.SAVE_DIR, "jak1_best_model.pth")):
                jak1_state = torch.load(os.path.join(Config.SAVE_DIR, "jak1_best_model.pth"))
                model.shared_layers.load_state_dict({
                    k.replace('shared_layers.', ''): v 
                    for k, v in jak1_state.items() 
                    if 'shared_layers' in k
                }, strict=False)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.INIT_LR * (2.0 if target == 'tyk2' else 1.0),
            weight_decay=Config.WEIGHT_DECAY
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=1,
            eta_min=Config.MIN_LR
        )
        
        criterion = AdaptiveLoss().to(Config.DEVICE) if target == 'tyk2' else nn.SmoothL1Loss()
        
        save_path = os.path.join(Config.SAVE_DIR, f"{target}_best_model.pth")
        best_val_r2 = train_model(
            model, train_loader, val_loader, 
            optimizer, scheduler, criterion,
            Config.N_EPOCHS, save_path,
            Config.DEVICE, Config.EARLY_STOP_PATIENCE,
            target
        )
        
        model.load_state_dict(torch.load(save_path))
        test_loss, test_r2 = evaluate(model, test_loader, Config.DEVICE, target)
        
        # 反标准化预测结果
        model.eval()
        with torch.no_grad():
            test_pred_normalized = model(torch.FloatTensor(X_test).to(Config.DEVICE), idx).cpu().numpy()
            test_pred_original = y_scaler.inverse_transform(test_pred_normalized.reshape(-1, 1)).flatten()
            
            # 直接从测试数据获取真实原始活性值
            test_true_original = test_data['activity'].values[:len(test_pred_original)]
            test_pred_binary = (test_pred_original >= Config.ACTIVITY_THRESHOLD).astype(int)
            
            # 保存结果
            result_df = pd.DataFrame({
                'SMILES': test_data['canonical_smiles'][:len(test_true_original)],
                'True_Activity': test_true_original,
                'Predicted_Activity': test_pred_original,
                'Predicted_Active': test_pred_binary,
                'Error': np.abs(test_true_original - test_pred_original)
            })
            result_df.to_csv(os.path.join(Config.SAVE_DIR, f"{target}_predictions.csv"), index=False)
        
        # 计算二分类指标
        binary_metrics = calculate_binary_metrics(
            test_true_original, 
            test_pred_original,
            threshold=Config.ACTIVITY_THRESHOLD
        )
        
        # 保存结果
        results[target] = {
            'test_r2': test_r2,
            'test_rmse': np.sqrt(test_loss),
            'best_val_r2': best_val_r2,
            'num_samples': len(train_data),
            **binary_metrics
        }
        
        # 绘制ROC曲线
        if binary_metrics['ROC_AUC'] is not np.nan:
            plot_roc_curve(
                y_true=(test_true_original >= Config.ACTIVITY_THRESHOLD).astype(int),
                y_score=test_pred_original,
                target_name=target,
                save_path=os.path.join(Config.SAVE_DIR, f"{target}_roc_curve.png")
            )
    
    # 保存和显示最终结果
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(Config.SAVE_DIR, "final_results.csv"))
    
    print("\n===== Final Results =====")
    print(results_df[['test_r2', 'test_rmse', 'best_val_r2', 'num_samples',
                      'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']])
    
    # 绘制结果
    plt.figure(figsize=(18, 8))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(results_df.index, results_df['test_r2'])
    plt.title('Test R² Scores', fontsize=12)
    plt.ylabel('R²', fontsize=10)
    plt.ylim(0.5, 0.9)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(results_df.index, results_df['test_rmse'])
    plt.title('Test RMSE', fontsize=12)
    plt.ylabel('RMSE', fontsize=10)
    plt.ylim(0.3, 0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.subplot(2, 2, 3)
    binary_metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1']
    bars = results_df[binary_metrics_cols].plot(kind='bar', ax=plt.gca())
    plt.title('Binary Classification Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=10)
    plt.ylim(0, 1.0)
    plt.legend(title='Metrics', bbox_to_anchor=(1, 1))
    for container in bars.containers:
        for bar in container:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(results_df.index, results_df['ROC_AUC'])
    plt.title('ROC AUC', fontsize=12)
    plt.ylabel('AUC', fontsize=10)
    plt.ylim(0.5, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, "performance_summary.png"), dpi=300)
    plt.close()
    
    # 绘制所有靶点的ROC曲线汇总图
    plt.figure(figsize=(10, 8))
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    for target in target_indices.keys():
        try:
            # 读取每个靶点的ROC数据
            result_df = pd.read_csv(os.path.join(Config.SAVE_DIR, f"{target}_predictions.csv"))
            y_true = (result_df['True_Activity'] >= Config.ACTIVITY_THRESHOLD).astype(int)
            y_score = result_df['Predicted_Activity']
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                     color=next(colors),
                     linestyle=next(linestyles),
                     linewidth=2,
                     label=f'{target.upper()} (AUC = {roc_auc:.2f})')
        except Exception as e:
            logging.warning(f"无法绘制{target}的ROC曲线: {str(e)}")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for KLSD Model', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('./roc_curves', "all_targets_roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()