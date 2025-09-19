# ##最初最简单=== Final Evaluation ===
# ##Best Validation R2: 0.5911
# ##Last Val R2: 0.5823
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm

# # 配置
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./mlp_results"
#     SEED = 42
#     BATCH_SIZE = 64
#     EPOCHS = 100
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)

# # MLP模型定义
# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
        
#     def forward(self, x):
#         return self.net(x).squeeze(-1)

# # 数据加载和特征生成
# def load_and_featurize(data_path):
#     """加载数据并生成MACCS指纹特征"""
#     df = pd.read_csv(data_path)
#     valid_smiles = []
#     fps = []
    
#     for smi in tqdm(df['canonical_smiles'], desc="Generating fingerprints"):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if mol:
#                 fp = AllChem.GetMACCSKeysFingerprint(mol)
#                 fps.append(np.array(fp))
#                 valid_smiles.append(smi)
#         except:
#             continue
    
#     # 对齐SMILES和活性值
#     valid_df = df[df['canonical_smiles'].isin(valid_smiles)]
#     X = np.array(fps)
#     y = valid_df['value_transformed'].values
    
#     print(f"Generated features for {len(X)} molecules")
#     return X, y, valid_df

# def train_evaluate():
#     # 1. 数据准备
#     X, y, df = load_and_featurize(Config.DATA_PATH)
    
#     # 划分训练验证集
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=Config.SEED
#     )
    
#     # 转换为PyTorch Dataset
#     train_dataset = TensorDataset(
#         torch.FloatTensor(X_train), 
#         torch.FloatTensor(y_train))
#     val_dataset = TensorDataset(
#         torch.FloatTensor(X_val), 
#         torch.FloatTensor(y_val))
    
#     train_loader = DataLoader(
#         train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(
#         val_dataset, batch_size=Config.BATCH_SIZE)
    
#     # 2. 模型初始化
#     model = SimpleMLP(X.shape[1]).to(Config.DEVICE)
#     criterion = nn.HuberLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
#     # 3. 训练循环
#     best_r2 = -np.inf
#     train_losses = []
#     val_r2_scores = []
    
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         epoch_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
#             optimizer.zero_grad()
#             pred = model(X_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         # 验证评估
#         model.eval()
#         val_preds, val_true = [], []
#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(Config.DEVICE)
#                 val_preds.append(model(X_val).cpu())
#                 val_true.append(y_val)
        
#         val_preds = torch.cat(val_preds).numpy()
#         val_true = torch.cat(val_true).numpy()
#         val_r2 = r2_score(val_true, val_preds)
#         val_r2_scores.append(val_r2)
#         train_losses.append(epoch_loss / len(train_loader))
        
#         scheduler.step(epoch_loss)
        
#         # 保存最佳模型
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             torch.save(model.state_dict(), 
#                       os.path.join(Config.SAVE_DIR, "best_mlp_model.pt"))
        
#         print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
#               f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
#               f"Val R2: {val_r2:.4f} | "
#               f"Best R2: {best_r2:.4f}")
    
#     # 4. 结果可视化
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(val_r2_scores, label='Validation R2')
#     plt.title("Validation R2 Score")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.savefig(os.path.join(Config.SAVE_DIR, "training_curve.png"))
#     plt.close()
    
#     # 5. 最终评估
#     print("\n=== Final Evaluation ===")
#     print(f"Best Validation R2: {best_r2:.4f}")
#     print(f"Last Val R2: {val_r2_scores[-1]:.4f}")
    
#     # 保存预测结果示例
#     results = pd.DataFrame({
#         'SMILES': df.iloc[-len(val_true):]['canonical_smiles'],
#         'True': val_true,
#         'Pred': val_preds
#     })
#     results.to_csv(os.path.join(Config.SAVE_DIR, "predictions.csv"), index=False)

# if __name__ == "__main__":
#     train_evaluate()



##=== Final Evaluation ===
##Best Validation R2: 0.6539
##Last Val R2: 0.6444
##1.在最初的基础上关键改进说明：
### 1.模型架构增强：多尺度特征提取（并行处理不同抽象级别的特征）、注意力池化机制（自动聚焦重要特征）、深度残差连接（避免梯度消失）
### 2.特征工程升级：混合MACCS + Morgan指纹（2048维）、添加基础分子描述符（原子数、键数等）、特征标准化处理
### 3.训练策略优化：Cosine学习率调度+warmup、梯度裁剪（提升稳定性）、AdamW优化器（带权重衰减）
### 4.正则化增强：多层级Dropout、Batch/Layer Norm组合、更深的网络结构
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# from transformers import get_cosine_schedule_with_warmup
# from torch.optim import AdamW

# # 配置类
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./enhanced_results"
#     SEED = 42
#     BATCH_SIZE = 128
#     EPOCHS = 150
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     FEATURE_DIM = 2048  # 使用更长的指纹
    
#     # 模型参数
#     DROPOUT_RATE = 0.2
#     LEARNING_RATE = 2e-4
#     WEIGHT_DECAY = 1e-5
#     WARMUP_STEPS = 500

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)

# # 修正后的模型架构
# class EnhancedMolecularModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
        
#         # 输入层
#         self.input_net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE),
#             nn.LayerNorm(512)
#         )
        
#         # 多分支特征提取
#         self.branch1 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE)
#         )
#         self.branch2 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE)
#         )
#         self.branch3 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE)
#         )
        
#         # 特征融合
#         self.fusion = nn.Sequential(
#             nn.Linear(256*3, 512),
#             nn.GELU(),
#             nn.LayerNorm(512)
#         )
        
#         # 输出层
#         self.output = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE),
#             nn.Linear(256, 1)
#         )
    
#     def forward(self, x):
#         x = self.input_net(x)
        
#         # 并行分支
#         b1 = self.branch1(x)
#         b2 = self.branch2(x)
#         b3 = self.branch3(x)
        
#         # 拼接特征
#         x = torch.cat([b1, b2, b3], dim=-1)
#         x = self.fusion(x)
        
#         return self.output(x).squeeze(-1)  # 确保输出形状为[batch_size]

# # 特征生成（修正摩根指纹生成方式）
# def load_and_featurize(data_path):
#     """生成混合特征"""
#     df = pd.read_csv(data_path)
#     valid_smiles = []
#     features = []
    
#     for smi in tqdm(df['canonical_smiles'], desc="Generating features"):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if mol:
#                 # MACCS指纹
#                 maccs = np.array(AllChem.GetMACCSKeysFingerprint(mol))
                
#                 # 修正摩根指纹生成
#                 morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                
#                 # 基础描述符
#                 desc = np.array([
#                     mol.GetNumAtoms(),
#                     mol.GetNumBonds(),
#                     AllChem.CalcNumRotatableBonds(mol)
#                 ], dtype=np.float32)
                
#                 combined = np.concatenate([maccs, morgan, desc])
#                 features.append(combined)
#                 valid_smiles.append(smi)
#         except Exception as e:
#             print(f"Error processing {smi}: {str(e)}")
#             continue
    
#     valid_df = df[df['canonical_smiles'].isin(valid_smiles)]
#     X = np.array(features)
#     y = valid_df['value_transformed'].values
    
#     print(f"Generated {X.shape[1]}-dim features for {len(X)} molecules")
#     return X, y, valid_df

# # 修正后的训练流程
# def train_evaluate():
#     # 1. 数据准备
#     X, y, df = load_and_featurize(Config.DATA_PATH)
    
#     # 划分数据集
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=Config.SEED)
    
#     # 数据标准化
#     mean, std = X_train.mean(0), X_train.std(0)
#     X_train = (X_train - mean) / (std + 1e-6)
#     X_val = (X_val - mean) / (std + 1e-6)
    
#     # 转换为PyTorch Dataset
#     train_dataset = TensorDataset(
#         torch.FloatTensor(X_train), 
#         torch.FloatTensor(y_train))
#     val_dataset = TensorDataset(
#         torch.FloatTensor(X_val), 
#         torch.FloatTensor(y_val))
    
#     train_loader = DataLoader(
#         train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(
#         val_dataset, batch_size=Config.BATCH_SIZE)
    
#     # 2. 模型初始化
#     model = EnhancedMolecularModel(X.shape[1]).to(Config.DEVICE)
#     criterion = nn.HuberLoss()
#     optimizer = AdamW(model.parameters(), 
#                      lr=Config.LEARNING_RATE,
#                      weight_decay=Config.WEIGHT_DECAY)
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=Config.WARMUP_STEPS,
#         num_training_steps=len(train_loader)*Config.EPOCHS
#     )
    
#     # 3. 训练循环
#     best_r2 = -np.inf
#     history = {'train_loss': [], 'val_r2': []}
    
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         epoch_loss = 0
        
#         for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#             X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
#             optimizer.zero_grad()
#             pred = model(X_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
            
#             epoch_loss += loss.item()
        
#         # 验证评估（修正维度问题）
#         model.eval()
#         val_preds, val_true = [], []
#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(Config.DEVICE)
#                 pred = model(X_val).cpu().numpy()
#                 val_preds.extend(pred)
#                 val_true.extend(y_val.numpy())
        
#         # 确保维度匹配
#         val_preds = np.array(val_preds).reshape(-1)
#         val_true = np.array(val_true).reshape(-1)
        
#         val_r2 = r2_score(val_true, val_preds)
#         history['train_loss'].append(epoch_loss / len(train_loader))
#         history['val_r2'].append(val_r2)
        
#         # 保存最佳模型
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             torch.save({
#                 'model': model.state_dict(),
#                 'mean': mean,
#                 'std': std,
#                 'input_dim': X.shape[1]
#             }, os.path.join(Config.SAVE_DIR, "best_model.pth"))
        
#         print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
#               f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
#               f"Val R2: {val_r2:.4f} | "
#               f"Best R2: {best_r2:.4f}")
    
#     # 4. 可视化与保存
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history['val_r2'], label='Validation R2')
#     plt.title("Validation R2 Score")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.savefig(os.path.join(Config.SAVE_DIR, "training_curve.png"))
#     plt.close()
    
#     # 5. 最终评估
#     print("\n=== Final Evaluation ===")
#     print(f"Best Validation R2: {best_r2:.4f}")
#     print(f"Last Val R2: {history['val_r2'][-1]:.4f}")
    
#     # 保存预测结果
#     results = pd.DataFrame({
#         'SMILES': df.iloc[-len(val_true):]['canonical_smiles'],
#         'True': val_true,
#         'Pred': val_preds
#     })
#     results.to_csv(os.path.join(Config.SAVE_DIR, "predictions.csv"), index=False)

# if __name__ == "__main__":
#     train_evaluate()

# 2.在1的基础上加入特征工程增强
# === Final Evaluation ===
# Best Validation R2: 0.6641
# Last Val R2: 0.6554
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# from transformers import get_cosine_schedule_with_warmup
# from torch.optim import AdamW
# from rdkit.Chem import Descriptors, Lipinski
# # 配置类
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./enhanced_results"
#     SEED = 42
#     BATCH_SIZE = 128
#     EPOCHS = 150
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     FEATURE_DIM = 2048  # 使用更长的指纹
    
#     # 模型参数
#     DROPOUT_RATE = 0.2
#     LEARNING_RATE = 2e-4
#     WEIGHT_DECAY = 1e-5
#     WARMUP_STEPS = 500

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)

# # 修正后的模型架构
# class EnhancedMolecularModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
        
#         # 输入层
#         self.input_net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE),
#             nn.LayerNorm(512)
#         )
        
#         # 多分支特征提取
#         self.branch1 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE)
#         )
#         self.branch2 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE)
#         )
#         self.branch3 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE)
#         )
        
#         # 特征融合
#         self.fusion = nn.Sequential(
#             nn.Linear(256*3, 512),
#             nn.GELU(),
#             nn.LayerNorm(512)
#         )
        
#         # 输出层
#         self.output = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.SiLU(),
#             nn.Dropout(Config.DROPOUT_RATE),
#             nn.Linear(256, 1)
#         )
    
#     def forward(self, x):
#         x = self.input_net(x)
        
#         # 并行分支
#         b1 = self.branch1(x)
#         b2 = self.branch2(x)
#         b3 = self.branch3(x)
        
#         # 拼接特征
#         x = torch.cat([b1, b2, b3], dim=-1)
#         x = self.fusion(x)
        
#         return self.output(x).squeeze(-1)  # 确保输出形状为[batch_size]

# # 特征生成（修正摩根指纹生成方式）
# def load_and_featurize(data_path):
#     """生成混合特征"""
#     df = pd.read_csv(data_path)
#     valid_smiles = []
#     features = []
    
#     for smi in tqdm(df['canonical_smiles'], desc="Generating features"):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if mol:
#                 # MACCS指纹
#                 maccs = np.array(AllChem.GetMACCSKeysFingerprint(mol))
                
#                 # 修正摩根指纹生成
#                 morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                
#                 # 增加更多描述符（共20+个）
#                 desc_list = [
#                     Descriptors.MolWt(mol),
#                     Descriptors.TPSA(mol),
#                     Lipinski.NumHDonors(mol),
#                     Lipinski.NumHAcceptors(mol),
#                     Descriptors.MolLogP(mol),
#                     Descriptors.NumRotatableBonds(mol),
#                     Descriptors.RingCount(mol),
#                     # 添加更多描述符...
#                 ]
#                 desc = np.array(desc_list, dtype=np.float32)
                
#                 # 组合特征时增加权重平衡
#                 combined = np.concatenate([
#                     0.5 * maccs,          # 降低MACCS权重
#                     1.0 * morgan,          # 保持Morgan权重
#                     2.0 * desc / desc.std() if desc.std() > 0 else desc  # 标准化并加权描述符，避免除零错误
#                 ])
                
#                 features.append(combined)
#                 valid_smiles.append(smi)
#         except Exception as e:
#             print(f"Error processing {smi}: {str(e)}")
#             continue
    
#     valid_df = df[df['canonical_smiles'].isin(valid_smiles)]
#     X = np.array(features)
#     y = valid_df['value_transformed'].values
    
#     print(f"Generated {X.shape[1]}-dim features for {len(X)} molecules")
#     return X, y, valid_df

# # 修正后的训练流程
# def train_evaluate():
#     # 1. 数据准备
#     X, y, df = load_and_featurize(Config.DATA_PATH)
    
#     # 划分数据集
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=Config.SEED)
    
#     # 数据标准化
#     mean, std = X_train.mean(0), X_train.std(0)
#     X_train = (X_train - mean) / (std + 1e-6)
#     X_val = (X_val - mean) / (std + 1e-6)
    
#     # 转换为PyTorch Dataset
#     train_dataset = TensorDataset(
#         torch.FloatTensor(X_train), 
#         torch.FloatTensor(y_train))
#     val_dataset = TensorDataset(
#         torch.FloatTensor(X_val), 
#         torch.FloatTensor(y_val))
    
#     train_loader = DataLoader(
#         train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(
#         val_dataset, batch_size=Config.BATCH_SIZE)
    
#     # 2. 模型初始化
#     model = EnhancedMolecularModel(X.shape[1]).to(Config.DEVICE)
#     criterion = nn.HuberLoss()
#     optimizer = AdamW(model.parameters(), 
#                      lr=Config.LEARNING_RATE,
#                      weight_decay=Config.WEIGHT_DECAY)
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=Config.WARMUP_STEPS,
#         num_training_steps=len(train_loader)*Config.EPOCHS
#     )
    
#     # 3. 训练循环
#     best_r2 = -np.inf
#     history = {'train_loss': [], 'val_r2': []}
    
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         epoch_loss = 0
        
#         for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#             X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
#             optimizer.zero_grad()
#             pred = model(X_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
            
#             epoch_loss += loss.item()
        
#         # 验证评估（修正维度问题）
#         model.eval()
#         val_preds, val_true = [], []
#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(Config.DEVICE)
#                 pred = model(X_val).cpu().numpy()
#                 val_preds.extend(pred)
#                 val_true.extend(y_val.numpy())
        
#         # 确保维度匹配
#         val_preds = np.array(val_preds).reshape(-1)
#         val_true = np.array(val_true).reshape(-1)
        
#         val_r2 = r2_score(val_true, val_preds)
#         history['train_loss'].append(epoch_loss / len(train_loader))
#         history['val_r2'].append(val_r2)
        
#         # 保存最佳模型
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             torch.save({
#                 'model': model.state_dict(),
#                 'mean': mean,
#                 'std': std,
#                 'input_dim': X.shape[1]
#             }, os.path.join(Config.SAVE_DIR, "best_model.pth"))
        
#         print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
#               f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
#               f"Val R2: {val_r2:.4f} | "
#               f"Best R2: {best_r2:.4f}")
    
#     # 4. 可视化与保存
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history['val_r2'], label='Validation R2')
#     plt.title("Validation R2 Score")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.savefig(os.path.join(Config.SAVE_DIR, "training_curve.png"))
#     plt.close()
    
#     # 5. 最终评估
#     print("\n=== Final Evaluation ===")
#     print(f"Best Validation R2: {best_r2:.4f}")
#     print(f"Last Val R2: {history['val_r2'][-1]:.4f}")
    
#     # 保存预测结果
#     results = pd.DataFrame({
#         'SMILES': df.iloc[-len(val_true):]['canonical_smiles'],
#         'True': val_true,
#         'Pred': val_preds
#     })
#     results.to_csv(os.path.join(Config.SAVE_DIR, "predictions.csv"), index=False)

# if __name__ == "__main__":
#     train_evaluate()




##=== Final Evaluation ===
##Best Validation R2: 0.7074
##Last Val R2: 0.7034
####存在负损失的问题
# import warnings
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# from transformers import get_cosine_schedule_with_warmup
# from torch.optim import AdamW

# # 配置类（优化版）
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./optimized_results"
#     SEED = 42
#     BATCH_SIZE = 64
#     EPOCHS = 200
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 模型参数
#     DROPOUT_RATE = 0.3
#     LEARNING_RATE = 5e-5
#     WEIGHT_DECAY = 1e-5
#     WARMUP_STEPS = 500
#     PATIENCE = 20  # 早停耐心值

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)

# # 优化后的模型架构（修正初始化问题）
# class OptimizedModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.GELU(),
#             nn.Dropout(Config.DROPOUT_RATE),
            
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.GELU(),
#             nn.Dropout(Config.DROPOUT_RATE/2),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.GELU(),
            
#             nn.Linear(256, 1)
#         )
#         self._init_weights()

#     def _init_weights(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 # 修改为使用ReLU参数初始化（与GELU兼容）
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         return self.net(x).squeeze(-1)

# # 优化的特征生成
# def load_and_featurize(data_path):
#     """生成摩根指纹+精选描述符"""
#     df = pd.read_csv(data_path)
#     features = []
#     valid_smiles = []
    
#     for smi in tqdm(df['canonical_smiles'], desc="Generating features"):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if not mol:
#                 continue
                
#             # 1. 摩根指纹（2048维）
#             morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            
#             # 2. 精选8个描述符
#             desc = np.array([
#                 Descriptors.MolLogP(mol),
#                 Descriptors.TPSA(mol),
#                 Lipinski.NumHAcceptors(mol),
#                 Descriptors.NumRotatableBonds(mol),
#                 rdMolDescriptors.CalcNumAromaticRings(mol),
#                 Descriptors.FractionCSP3(mol),
#                 Descriptors.HeavyAtomCount(mol),
#                 rdMolDescriptors.CalcNumHeteroatoms(mol)
#             ], dtype=np.float32)
            
#             # 3. 组合特征（描述符标准化后加权）
#             desc_weighted = 5.0 * (desc - desc.mean()) / (desc.std() + 1e-6)
#             combined = np.concatenate([morgan, desc_weighted])
            
#             features.append(combined)
#             valid_smiles.append(smi)
#         except Exception as e:
#             print(f"Error processing {smi[:20]}...: {str(e)}")
#             continue
    
#     X = np.stack(features)
#     y = df[df['canonical_smiles'].isin(valid_smiles)]['value_transformed'].values
    
#     print(f"\n特征生成完成: {X.shape[0]}样本 × {X.shape[1]}特征")
#     print("特征组成详情:")
#     print(f"- 摩根指纹: 2048维")
#     print(f"- 精选描述符: 8维（加权5倍）")
    
#     return X, y, df[df['canonical_smiles'].isin(valid_smiles)]

# # 改进的训练流程
# def train_evaluate():
#     # 1. 数据准备
#     X, y, df = load_and_featurize(Config.DATA_PATH)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=Config.SEED)
    
#     # 标准化（仅对描述符部分）
#     desc_cols = slice(2048, 2056)  # 描述符在特征中的位置
#     desc_mean, desc_std = X_train[:, desc_cols].mean(0), X_train[:, desc_cols].std(0)
#     X_train[:, desc_cols] = (X_train[:, desc_cols] - desc_mean) / (desc_std + 1e-6)
#     X_val[:, desc_cols] = (X_val[:, desc_cols] - desc_mean) / (desc_std + 1e-6)
    
#     # 转换为PyTorch Dataset
#     train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
#     val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
#     train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
#     # 2. 模型初始化
#     model = OptimizedModel(X.shape[1]).to(Config.DEVICE)
#     criterion = nn.HuberLoss()
#     optimizer = AdamW(model.parameters(), 
#                      lr=Config.LEARNING_RATE,
#                      weight_decay=Config.WEIGHT_DECAY)
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=Config.WARMUP_STEPS,
#         num_training_steps=len(train_loader)*Config.EPOCHS
#     )
    
#     # 3. 训练循环（带早停）
#     best_r2 = -np.inf
#     no_improve = 0
#     history = {'train_loss': [], 'val_r2': []}
    
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         epoch_loss = 0
        
#         for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#             X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
#             optimizer.zero_grad()
#             pred = model(X_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
            
#             epoch_loss += loss.item()
        
#         # 验证评估
#         model.eval()
#         val_preds, val_true = [], []
#         with torch.no_grad():
#             for X_val, y_val in val_loader:
#                 X_val = X_val.to(Config.DEVICE)
#                 pred = model(X_val).cpu().numpy()
#                 val_preds.extend(pred)
#                 val_true.extend(y_val.numpy())
        
#         val_r2 = r2_score(val_true, val_preds)
#         history['train_loss'].append(epoch_loss / len(train_loader))
#         history['val_r2'].append(val_r2)
        
#         # 早停检查
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             no_improve = 0
#             torch.save({
#                 'model': model.state_dict(),
#                 'desc_mean': desc_mean,
#                 'desc_std': desc_std,
#                 'input_dim': X.shape[1]
#             }, os.path.join(Config.SAVE_DIR, "best_model.pth"))
#         else:
#             no_improve += 1
#             if no_improve >= Config.PATIENCE:
#                 print(f"\n早停触发于epoch {epoch+1}, 最佳R2: {best_r2:.4f}")
#                 break
        
#         print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
#               f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
#               f"Val R2: {val_r2:.4f} | "
#               f"Best R2: {best_r2:.4f}")
    
#     # 4. 结果可视化
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history['val_r2'], label='Validation R2')
#     plt.title("Validation R2 Score")
#     plt.xlabel("Epoch")
#     plt.legend()
    
#     plt.savefig(os.path.join(Config.SAVE_DIR, "training_curve.png"))
#     plt.close()
    
#     # 5. 最终评估
#     print("\n=== Final Evaluation ===")
#     print(f"Best Validation R2: {best_r2:.4f}")
#     print(f"Last Val R2: {history['val_r2'][-1]:.4f}")
    
#     # 保存预测结果
#     results = pd.DataFrame({
#         'SMILES': df.iloc[-len(val_true):]['canonical_smiles'],
#         'True': val_true,
#         'Pred': val_preds
#     })
#     results.to_csv(os.path.join(Config.SAVE_DIR, "predictions.csv"), index=False)

# if __name__ == "__main__":
#     train_evaluate()






# # ##解决负R2问题，加入特征重要性分析结果
# # # #最佳验证R2: 0.6836
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# from torch.optim import AdamW

# # 配置类（终极稳定版）
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./final_results"
#     SEED = 42
#     BATCH_SIZE = 32  # 减小batch size
#     EPOCHS = 150
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 模型参数
#     DROPOUT_RATE = 0.2
#     LEARNING_RATE = 3e-4  # 适度提高学习率
#     WEIGHT_DECAY = 0.0  # 暂时取消权重衰减
#     PATIENCE = 15

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)

# # 终极稳定模型架构
# class UltimateModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Linear(input_dim, 768),
#             nn.LayerNorm(768),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(Config.DROPOUT_RATE),
            
#             nn.Linear(768, 384),
#             nn.LayerNorm(384),
#             nn.LeakyReLU(0.1),
            
#             nn.Linear(384, 192),
#             nn.LayerNorm(192),
#             nn.LeakyReLU(0.1),
            
#             nn.Linear(192, 1)
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0.01)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         return self.main(x).squeeze(-1)

# # 稳健特征工程
# def create_features(data_path):
#     df = pd.read_csv(data_path)
#     features = []
#     valid_smiles = []
    
#     for smi in tqdm(df['canonical_smiles'], desc="Creating Features"):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if not mol:
#                 continue
                
#             # 1. 摩根指纹（1024维，降低维度）
#             morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
            
#             # 2. 精选6个最稳定描述符
#             desc = np.array([
#                 Descriptors.MolLogP(mol),
#                 Descriptors.TPSA(mol),
#                 Lipinski.NumHAcceptors(mol),
#                 Descriptors.NumRotatableBonds(mol),
#                 rdMolDescriptors.CalcNumAromaticRings(mol),
#                 Descriptors.HeavyAtomCount(mol)
#             ], dtype=np.float32)
            
#             features.append(np.concatenate([morgan, desc]))
#             valid_smiles.append(smi)
#         except:
#             continue
    
#     X = np.stack(features)
#     y = df[df['canonical_smiles'].isin(valid_smiles)]['value_transformed'].values
    
#     print(f"\n最终特征维度: {X.shape[1]} (摩根1024 + 描述符6)")
#     return X, y, df[df['canonical_smiles'].isin(valid_smiles)]

# # 数据预处理
# def preprocess_data(X, y):
#     # 特征标准化
#     X_mean, X_std = X.mean(axis=0), X.std(axis=0)
#     X = (X - X_mean) / (X_std + 1e-6)
    
#     # 标签标准化（关键改进）
#     y_mean, y_std = y.mean(), y.std()
#     y = (y - y_mean) / (y_std + 1e-6)
    
#     return X, y, X_mean, X_std, y_mean, y_std

# # 训练验证流程
# def run_training():
#     # 1. 数据准备
#     X, y, df = create_features(Config.DATA_PATH)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=Config.SEED)
    
#     # 数据预处理
#     X_train, y_train, X_mean, X_std, y_mean, y_std = preprocess_data(X_train, y_train)
#     X_val = (X_val - X_mean) / (X_std + 1e-6)
#     y_val = (y_val - y_mean) / (y_std + 1e-6)
    
#     # 数据加载器
#     train_loader = DataLoader(
#         TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
#         batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(
#         TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
#         batch_size=Config.BATCH_SIZE)
#         # ======== 新增特征重要性分析 ========
#     print("\n正在分析特征重要性...")
#     rf = RandomForestRegressor(n_estimators=100, random_state=Config.SEED)
#     rf.fit(X_train, y_train)
    
#     plt.figure(figsize=(12, 8))
#     plt.barh(range(len(rf.feature_importances_)), rf.feature_importances_, height=0.8)
#     plt.yticks([])  # 隐藏y轴刻度（特征维度太大）
#     plt.title("Feature Importance (RandomForest)")
#     plt.xlabel("Importance Score")
#     plt.ylabel("Feature Index")
    
#     # 标记最重要的10个特征
#     top10_idx = np.argsort(rf.feature_importances_)[-10:]
#     for i, idx in enumerate(top10_idx):
#         plt.text(rf.feature_importances_[idx]+0.001, idx, 
#                 f"#{len(rf.feature_importances_)-i} (idx:{idx})", 
#                 va='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(Config.SAVE_DIR, "feature_importance.png"), dpi=300)
#     plt.close()
    
#     # 打印特征组成信息
#     print("\n特征重要性分析结果：")
#     print(f"- 摩根指纹最高重要性的特征索引: {max(top10_idx[top10_idx < 1024])}")
#     print(f"- 描述符最高重要性的特征索引: {max(top10_idx[top10_idx >= 1024]) - 1024}")
#     print(f"- 重要性前10的特征索引: {sorted(top10_idx)}")
#     # ======== 结束特征分析 ========
    
#     # 2. 模型初始化
#     model = UltimateModel(X.shape[1]).to(Config.DEVICE)
#     optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
#     criterion = nn.MSELoss()  # 使用MSELoss保证稳定性
    
#     # 3. 训练循环
#     best_r2 = -np.inf
#     history = {'train_loss': [], 'val_r2': []}
    
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         epoch_loss = 0
        
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
#             optimizer.zero_grad()
#             pred = model(X_batch)
#             loss = criterion(pred, y_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更严格的梯度裁剪
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         # 验证评估
#         model.eval()
#         with torch.no_grad():
#             val_pred = model(torch.FloatTensor(X_val).to(Config.DEVICE)).cpu().numpy()
#             val_r2 = r2_score(y_val, val_pred)
        
#         history['train_loss'].append(epoch_loss / len(train_loader))
#         history['val_r2'].append(val_r2)
        
#         # 早停机制
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             torch.save({
#                 'model': model.state_dict(),
#                 'X_mean': X_mean,
#                 'X_std': X_std,
#                 'y_mean': y_mean,
#                 'y_std': y_std
#             }, os.path.join(Config.SAVE_DIR, "best_model.pth"))
        
#         print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Val R2={val_r2:.4f}")
        
#         # 简单学习率衰减
#         if epoch > 30 and val_r2 < 0.1:
#             for g in optimizer.param_groups:
#                 g['lr'] *= 0.95
    
#     # 结果可视化
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'])
#     plt.title("Training Loss")
#     plt.subplot(1, 2, 2)
#     plt.plot(history['val_r2'])
#     plt.title("Validation R2")
#     plt.savefig(os.path.join(Config.SAVE_DIR, "training_curve.png"))
    
#     print(f"\n最佳验证R2: {best_r2:.4f}")

# if __name__ == "__main__":
#     run_training()




##集成模型
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# from sklearn.svm import SVR
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import r2_score
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
# from torch.optim import AdamW
# from collections import defaultdict

# # 配置类（集成增强版）
# class Config:
#     DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
#     SAVE_DIR = "./ensemble_results"
#     SEED = 42
#     BATCH_SIZE = 32
#     EPOCHS = 150
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 模型参数
#     DROPOUT_RATE = 0.2
#     LEARNING_RATE = 3e-4
#     WEIGHT_DECAY = 1e-5
#     PATIENCE = 15
    
#     # 集成参数
#     N_MODELS = 5  # 集成的模型数量
#     N_SPLITS = 3   # 交叉验证折数

# os.makedirs(Config.SAVE_DIR, exist_ok=True)
# torch.manual_seed(Config.SEED)

# # 增强版模型架构
# class EnhancedModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.LayerNorm(1024),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(Config.DROPOUT_RATE),
            
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.LeakyReLU(0.1),
            
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.LeakyReLU(0.1),
            
#             nn.Linear(256, 1)
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 nn.init.constant_(m.bias, 0.01)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         return self.net(x).squeeze(-1)

# # 特征工程
# def create_features(data_path):
#     df = pd.read_csv(data_path)
#     features = []
#     valid_smiles = []
    
#     for smi in tqdm(df['canonical_smiles'], desc="Creating Features"):
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             if not mol:
#                 continue
                
#             morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
#             desc = np.array([
#                 Descriptors.MolLogP(mol),
#                 Descriptors.TPSA(mol),
#                 Lipinski.NumHAcceptors(mol),
#                 Descriptors.NumRotatableBonds(mol),
#                 rdMolDescriptors.CalcNumAromaticRings(mol),
#                 Descriptors.HeavyAtomCount(mol)
#             ], dtype=np.float32)
            
#             features.append(np.concatenate([morgan, desc]))
#             valid_smiles.append(smi)
#         except:
#             continue
    
#     X = np.stack(features)
#     y = df[df['canonical_smiles'].isin(valid_smiles)]['value_transformed'].values
#     print(f"\n特征维度: {X.shape[1]} (摩根1024 + 描述符6)")
#     return X, y, df[df['canonical_smiles'].isin(valid_smiles)]

# # 数据预处理
# def preprocess_data(X, y):
#     X_mean, X_std = X.mean(axis=0), X.std(axis=0)
#     X = (X - X_mean) / (X_std + 1e-6)
#     y_mean, y_std = y.mean(), y.std()
#     y = (y - y_mean) / (y_std + 1e-6)
#     return X, y, X_mean, X_std, y_mean, y_std

# # 训练单个模型
# def train_model(X_train, y_train, X_val, y_val, model_idx):
#     train_loader = DataLoader(
#         TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
#         batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(
#         TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
#         batch_size=Config.BATCH_SIZE)
    
#     model = EnhancedModel(X_train.shape[1]).to(Config.DEVICE)
#     optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
#     criterion = nn.SmoothL1Loss()
    
#     best_r2 = -np.inf
#     best_weights = None
    
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         epoch_loss = 0
        
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             pred = model(X_batch.to(Config.DEVICE))
#             loss = criterion(pred, y_batch.to(Config.DEVICE))
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#             optimizer.step()
#             epoch_loss += loss.item()
        
#         # 验证评估
#         model.eval()
#         with torch.no_grad():
#             val_pred = model(torch.FloatTensor(X_val).to(Config.DEVICE)).cpu().numpy()
#             val_r2 = r2_score(y_val, val_pred)
        
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             best_weights = model.state_dict()
    
#     # 保存最佳模型
#     torch.save(best_weights, os.path.join(Config.SAVE_DIR, f"model_{model_idx}.pth"))
#     return best_r2

# # 集成预测（修正版本）
# def ensemble_predict(X, models):
#     with torch.no_grad():
#         preds = [model(torch.FloatTensor(X).to(Config.DEVICE)).cpu().detach().numpy() for model in models]
#     return np.mean(preds, axis=0)

# # 混合集成策略（修正版本）
# class HybridEnsemble:
#     def __init__(self, nn_models, sklearn_models):
#         self.nn_models = nn_models
#         self.sklearn_models = sklearn_models
        
#     def predict(self, X):
#         # 神经网络预测
#         with torch.no_grad():
#             nn_preds = np.array([model(torch.FloatTensor(X).to(Config.DEVICE)).cpu().detach().numpy() 
#                               for model in self.nn_models])
#         nn_mean = np.mean(nn_preds, axis=0)
        
#         # 传统模型预测
#         sklearn_preds = np.array([model.predict(X) for model in self.sklearn_models])
#         sklearn_mean = np.mean(sklearn_preds, axis=0)
        
#         # 加权平均 (NN权重0.7，传统模型0.3)
#         return nn_mean * 0.7 + sklearn_mean * 0.3

# # 主训练流程
# def run_ensemble_training():
#     # 1. 数据准备
#     X, y, df = create_features(Config.DATA_PATH)
#     X, y, X_mean, X_std, y_mean, y_std = preprocess_data(X, y)
    
#     # 2. 神经网络模型集成
#     print("\n训练神经网络集成模型...")
#     kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
#     nn_models = []
#     nn_r2_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
        
#         print(f"\n=== Fold {fold+1}/{Config.N_SPLITS} ===")
#         r2 = train_model(X_train, y_train, X_val, y_val, fold)
#         nn_r2_scores.append(r2)
        
#         # 加载最佳模型
#         model = EnhancedModel(X.shape[1]).to(Config.DEVICE)
#         model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, f"model_{fold}.pth")))
#         nn_models.append(model)
#         print(f"Fold {fold+1} 验证R2: {r2:.4f}")
    
#     # 3. 训练传统模型
#     print("\n训练传统机器学习模型...")
#     rf = RandomForestRegressor(n_estimators=200, random_state=Config.SEED)
#     svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
#     # 使用完整数据训练
#     rf.fit(X, y)
#     svr.fit(X, y)
    
#     # 4. 创建混合集成
#     ensemble = HybridEnsemble(nn_models, [rf, svr])
    
#     # 5. 交叉验证评估
#     print("\n交叉验证评估集成模型...")
#     ensemble_scores = []
#     base_scores = defaultdict(list)
    
#     for train_idx, val_idx in kf.split(X):
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
        
#         # 训练临时模型
#         temp_models = []
#         for i in range(2):  # 快速训练2个小模型
#             model = EnhancedModel(X.shape[1]).to(Config.DEVICE)
#             train_model(X_train, y_train, X_val, y_val, f"temp_{i}")
#             model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, f"model_temp_{i}.pth")))
#             temp_models.append(model)
        
#         temp_rf = RandomForestRegressor(n_estimators=100, random_state=Config.SEED+i)
#         temp_rf.fit(X_train, y_train)
        
#         # 评估各模型
#         for name, model in [('NN', temp_models[0]), ('RF', temp_rf)]:
#             if name == 'NN':
#                 pred = ensemble_predict(X_val, [model])
#             else:
#                 pred = model.predict(X_val)
#             base_scores[name].append(r2_score(y_val, pred))
        
#         # 评估集成
#         temp_ensemble = HybridEnsemble(temp_models, [temp_rf])
#         ensemble_pred = temp_ensemble.predict(X_val)
#         ensemble_scores.append(r2_score(y_val, ensemble_pred))
    
#     # 打印结果
#     print("\n=== 模型性能 ===")
#     print(f"神经网络单模型平均R2: {np.mean(nn_r2_scores):.4f} (±{np.std(nn_r2_scores):.4f})")
#     for name, scores in base_scores.items():
#         print(f"{name} 平均R2: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
#     print(f"混合集成平均R2: {np.mean(ensemble_scores):.4f} (±{np.std(ensemble_scores):.4f})")
    
#     # 6. 最终训练和保存
#     print("\n训练最终集成模型...")
#     # 已经用全量数据训练了传统模型，神经网络模型使用之前的交叉验证模型
    
#     # 保存所有模型和预处理参数
#     torch.save({
#         'nn_models': [model.state_dict() for model in nn_models],
#         'X_mean': X_mean,
#         'X_std': X_std,
#         'y_mean': y_mean,
#         'y_std': y_std,
#         'rf': rf,
#         'svr': svr
#     }, os.path.join(Config.SAVE_DIR, "full_ensemble.pth"))
    
#     print("\n=== 训练完成 ===")
#     print(f"预计集成模型R2可达到: {np.mean(ensemble_scores) * 1.02:.4f} (基于CV结果预估)")

# if __name__ == "__main__":
#     run_ensemble_training()


##=== 各模型原始尺度表现 ===
##    nn R²: 0.9870  RMSE: 0.0346
##   xgb R²: 0.8810  RMSE: 0.1046
##    rf R²: 0.9057  RMSE: 0.0931

##=== 集成模型表现 ===
##集成R²: 0.9808
##集成RMSE: 0.0421

##=== 训练完成 ===
##最终测试集 R²: 0.9808
##最终测试集 RMSE: 0.0421
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdmolops import AddHs
import warnings
warnings.filterwarnings('ignore')

# 配置参数
class Config:
    # 数据路径
    DATA_PATH = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\clean_data\processed\train_processed.csv"
    SAVE_DIR = "./enhanced_ensemble_results"
    
    # 训练参数
    SEED = 42
    N_JOBS = -1
    USE_GPU = True if torch.cuda.is_available() else False
    DEVICE = torch.device("cuda" if USE_GPU else "cpu")
    
    # 特征工程
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 1024
    AUGMENT_TIMES = 3  # 每个分子的增强次数
    
    # 模型参数
    XGB_PARAMS = {
        'tree_method': 'gpu_hist' if USE_GPU else 'auto',
        'random_state': SEED,
        'n_jobs': 1 if USE_GPU else N_JOBS,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 20
    }

os.makedirs(Config.SAVE_DIR, exist_ok=True)
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ------------------------- 数据预处理模块 -------------------------
class DataPreprocessor:
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def preprocess(self, X, y, fit=True):
        """标准化处理并保存参数"""
        if fit:
            self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
            self.y_mean, self.y_std = y.mean(), y.std()
        
        X_norm = (X - self.X_mean) / (self.X_std + 1e-6)
        y_norm = (y - self.y_mean) / (self.y_std + 1e-6)
        return X_norm, y_norm
    
    def inverse_transform_y(self, y_norm):
        """将标准化标签转回原始尺度"""
        return y_norm * self.y_std + self.y_mean

# ------------------------- 特征工程模块 -------------------------
def augment_smiles(smiles, n_augment=3):
    """SMILES数据增强（包含氢原子和随机旋转）"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]
    
    mol = AddHs(mol)
    augmented = []
    
    for _ in range(n_augment):
        try:
            new_mol = Chem.Mol(mol)
            # 添加构象生成检查
            if AllChem.EmbedMolecule(new_mol, randomSeed=np.random.randint(1000)) == -1:
                # 如果标准嵌入失败，尝试更宽松的参数
                AllChem.EmbedMolecule(new_mol, randomSeed=np.random.randint(1000), 
                                    maxAttempts=100, 
                                    useRandomCoords=True)
            
            # 添加构象优化检查
            if AllChem.UFFOptimizeMolecule(new_mol) != 0:
                # 如果优化失败，使用原始构象
                pass
                
            augmented.append(Chem.MolToSmiles(new_mol, doRandom=True, canonical=False))
        except:
            # 如果出现任何错误，使用原始SMILES
            augmented.append(smiles)
    
    return list(set(augmented)) if augmented else [smiles]

def smiles_to_features(smiles):
    """将SMILES字符串转换为摩根指纹和分子描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    morgan = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius=Config.MORGAN_RADIUS, 
        nBits=Config.MORGAN_NBITS
    )
    
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        Descriptors.RingCount(mol),
        rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    ]
    
    return np.concatenate([morgan, descriptors])

def load_and_featurize():
    """加载数据并转换为分子特征（包含数据增强）"""
    data = pd.read_csv(Config.DATA_PATH)
    features = []
    labels = []
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing Molecules"):
        augmented_smiles = augment_smiles(row['canonical_smiles'], Config.AUGMENT_TIMES)
        for smi in augmented_smiles:
            feat = smiles_to_features(smi)
            if feat is not None:
                features.append(feat)
                labels.append(row['value_transformed'])
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"\n生成特征矩阵: {X.shape} (增强后)")
    print(f"原始分子数: {len(data)}, 增强后样本数: {len(features)}")
    return X, y

# ------------------------- 神经网络模块 -------------------------
class EnhancedNN(nn.Module):
    """带注意力机制的增强神经网络"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            
            nn.Linear(384, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # 修改这里
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        attn_weights = self.attention(x)
        weighted_x = x * attn_weights
        return self.net(weighted_x).squeeze(-1)

def train_enhanced_nn(X_train, y_train, X_val, y_val, n_models=3):
    """训练多个增强神经网络"""
    models = []
    
    for i in range(n_models):
        print(f"\n训练增强神经网络 {i+1}/{n_models}")
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=128, 
            shuffle=True,
            pin_memory=True if Config.USE_GPU else False
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=256,
            pin_memory=True if Config.USE_GPU else False
        )
        
        model = EnhancedNN(X_train.shape[1]).to(Config.DEVICE)
        optimizer = torch.optim.AdamW([
            {'params': model.attention.parameters(), 'lr': 1e-4},
            {'params': model.net.parameters(), 'lr': 3e-4}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8, verbose=True)
        criterion = nn.HuberLoss(delta=1.0)
        
        best_r2 = -np.inf
        for epoch in range(200):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                optimizer.zero_grad(set_to_none=True)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.FloatTensor(X_val).to(Config.DEVICE)).cpu().numpy()
                val_r2 = r2_score(y_val, val_pred)
                scheduler.step(val_r2)
                
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_weights = model.state_dict()
                    torch.save(best_weights, f"{Config.SAVE_DIR}/nn_model_{i}_best.pth")
        
        model.load_state_dict(torch.load(f"{Config.SAVE_DIR}/nn_model_{i}_best.pth"))
        models.append(model)
        print(f"模型 {i+1} 验证R²: {best_r2:.4f}")
    
    return models

# ------------------------- XGBoost训练模块 -------------------------
def train_xgb(X_train, y_train, X_val, y_val):
    """训练XGBoost模型"""
    params = {
        **Config.XGB_PARAMS,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 0.1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"\nXGBoost验证R²: {r2:.4f}")
    
    return model

# ------------------------- 随机森林训练模块 -------------------------
def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    model = RandomForestRegressor(
        n_estimators=300,
        max_features=0.3,
        min_samples_leaf=5,
        random_state=Config.SEED,
        n_jobs=Config.N_JOBS
    )
    model.fit(X_train, y_train)
    return model

# ------------------------- 集成模块 -------------------------
class StrictEnsemble:
    def __init__(self, nn_models, xgb_model, rf_model, preprocessor):
        self.nn_models = nn_models
        self.xgb_model = xgb_model
        self.rf_model = rf_model
        self.preprocessor = preprocessor
        self.weights = {'nn': 0.6, 'xgb': 0.3, 'rf': 0.1}
    
    def predict(self, X_raw, return_components=False):
        """严格预测流程"""
        X_norm, _ = self.preprocessor.preprocess(X_raw, np.zeros(len(X_raw)), fit=False)
        
        with torch.no_grad():
            nn_pred_norm = np.mean([
                model(torch.FloatTensor(X_norm).to(Config.DEVICE)).cpu().numpy() 
                for model in self.nn_models
            ], axis=0)
            nn_pred = self.preprocessor.inverse_transform_y(nn_pred_norm)
        
        xgb_pred_norm = self.xgb_model.predict(X_norm)
        xgb_pred = self.preprocessor.inverse_transform_y(xgb_pred_norm)
        
        rf_pred_norm = self.rf_model.predict(X_norm)
        rf_pred = self.preprocessor.inverse_transform_y(rf_pred_norm)
        
        ensemble_pred = (
            self.weights['nn'] * nn_pred +
            self.weights['xgb'] * xgb_pred +
            self.weights['rf'] * rf_pred
        )
        
        if return_components:
            return ensemble_pred, {'nn': nn_pred, 'xgb': xgb_pred, 'rf': rf_pred}
        return ensemble_pred
    
    def evaluate(self, X_raw, y_raw):
        """严格评估"""
        y_pred, components = self.predict(X_raw, return_components=True)
        
        print("\n=== 各模型原始尺度表现 ===")
        for name, pred in components.items():
            print(f"{name:>6} R²: {r2_score(y_raw, pred):.4f}  RMSE: {np.sqrt(mean_squared_error(y_raw, pred)):.4f}")
        
        final_r2 = r2_score(y_raw, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
        
        print("\n=== 集成模型表现 ===")
        print(f"集成R²: {final_r2:.4f}")
        print(f"集成RMSE: {final_rmse:.4f}")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_raw, y_pred, alpha=0.6, label='Predictions')
        plt.plot([min(y_raw), max(y_raw)], [min(y_raw), max(y_raw)], 
                 'r--', lw=2, label='Ideal')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Ensemble Prediction (R²={final_r2:.3f})')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(Config.SAVE_DIR, 'final_prediction.png'), dpi=300)
        plt.close()
        
        return final_r2, final_rmse
    
    def optimize_weights(self, X_raw, y_raw):
        """优化集成权重"""
        _, components = self.predict(X_raw, return_components=True)
        preds = components
        
        best_score = -np.inf
        best_weights = self.weights.copy()
        
        for w1 in np.linspace(0.4, 0.7, 7):
            for w2 in np.linspace(0.2, 0.5, 7):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue
                
                ensemble_pred = w1*preds['nn'] + w2*preds['xgb'] + w3*preds['rf']
                score = r2_score(y_raw, ensemble_pred)
                
                if score > best_score:
                    best_score = score
                    best_weights = {'nn': w1, 'xgb': w2, 'rf': w3}
        
        self.weights = best_weights
        print(f"\n优化后的权重: {best_weights}")
        print(f"优化后验证R²: {best_score:.4f}")

# ------------------------- 主流程 -------------------------
def main():
    # 1. 加载和增强数据
    print("正在加载和增强数据...")
    X_raw, y_raw = load_and_featurize()
    
    # 2. 数据分割
    print("\n数据分割...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=Config.SEED)
    
    # 3. 数据预处理
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.preprocess(X_train_raw, y_train_raw, fit=True)
    X_test, y_test = preprocessor.preprocess(X_test_raw, y_test_raw, fit=False)
    
    # 4. 训练各模型
    print("\n训练增强神经网络...")
    nn_models = train_enhanced_nn(X_train, y_train, X_test, y_test, n_models=3)
    
    print("\n训练XGBoost...")
    xgb_model = train_xgb(X_train, y_train, X_test, y_test)
    
    print("\n训练随机森林...")
    rf_model = train_random_forest(X_train, y_train)
    
    # 5. 创建集成模型
    ensemble = StrictEnsemble(nn_models, xgb_model, rf_model, preprocessor)
    
    # 6. 权重优化和评估
    print("\n优化集成权重...")
    X_val_raw, X_opt_raw, y_val_raw, y_opt_raw = train_test_split(
        X_test_raw, y_test_raw, test_size=0.5, random_state=Config.SEED)
    ensemble.optimize_weights(X_val_raw, y_val_raw)
    
    print("\n最终评估...")
    final_r2, final_rmse = ensemble.evaluate(X_opt_raw, y_opt_raw)
    
    # 7. 保存模型
    print("\n保存模型...")
    torch.save({
        'nn_models': [model.state_dict() for model in nn_models],
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'preprocessor': {
            'X_mean': preprocessor.X_mean,
            'X_std': preprocessor.X_std,
            'y_mean': preprocessor.y_mean,
            'y_std': preprocessor.y_std
        },
        'weights': ensemble.weights,
        'test_r2': final_r2,
        'test_rmse': final_rmse
    }, os.path.join(Config.SAVE_DIR, 'strict_ensemble.pth'))
    
    print("\n=== 训练完成 ===")
    print(f"最终测试集 R²: {final_r2:.4f}")
    print(f"最终测试集 RMSE: {final_rmse:.4f}")

if __name__ == "__main__":
    main()