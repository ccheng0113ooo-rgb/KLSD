# import os
# from tqdm import tqdm
# import math
# # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from CNNforclassification import CNNforclassification
# from smiles2graph import gen_smiles2graph
# import pandas as pd
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import KFold,train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
# from const import *
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
#
# class jak_dataset(Dataset):
#     def __init__(self, dataframe, max_len=80):
#         super(jak_dataset, self).__init__()
#         self.len = len(dataframe)
#         self.dataframe = dataframe
#         self.max_len = max_len
#
#     def __getitem__(self, idx):
#         X = torch.zeros(self.max_len)
#         y = 1 if self.dataframe.Activity[idx] == 1 else 0
#         for idx, atom in enumerate(list(self.dataframe.Smiles[idx])[:self.max_len]):
#             X[idx] = vocabulary[atom]
#
#         return X.long(), y
#
#     def __len__(self):
#         return self.len
#
#
# if __name__ == '__main__':
#     # jak_number = int(cuda_num / 2 + 1)
#     # jak_number=1
#     # jak_df = pd.read_csv('jak/final_JAK'+str(jak_number)+'.csv',index_col=0).reset_index(drop=True)
# # ####原始文件读入
# #     jak_df = pd.read_csv('jak/JAK' + str(jak_number) + '_final.csv')
#     # jak_df = pd.read_csv('jak/final_JAK' + str(jak_number) + '.csv')
#     # 导入 pandas 的 read_excel 方法来读取 Excel 文件
#     file_path = 'jak/final_JAK1.xlsx'
#     jak_number = int(file_path.split('_JAK')[1].split('.')[0])  # 提取 "1"
#     # 读取 Excel 文件
#     jak_df = pd.read_excel(file_path)
#
#     # jak1_small_df = pd.read_csv('JAK'+str(jak_number)+'_active.csv').drop(columns='Unnamed: 0')
#     # jak1_small_df.columns = jak1_df.columns
#     # jak1_small_df['Activity'] = 1 - jak1_small_df['Activity']
#     # jak_df = pd.concat([jak1_df, jak1_small_df]).reset_index(drop=True).drop_duplicates().reset_index(drop=True)
#     print(jak_df)
#
#     weight_dict = {1: torch.tensor([3.0, 1.0]), 2: torch.tensor([2.0, 1.0]), 3: torch.tensor([1.0, 1.0]),
#                    4: torch.tensor([2.0, 1.0])}
#
#     # k = 5
#     # kfold = KFold(n_splits=k, shuffle=True,random_state=42)
#     params = {'batch_size': 16, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
#     epoches = 20
#
#     known_df = pd.DataFrame(known_drugs)
#     known_df.columns=['Smiles']
#     known_df['Activity']=0
#     known_data=jak_dataset(known_df)
#     known_loader=DataLoader(known_data,**params)
#     print(known_df)
#
#     acc_know_pred = torch.zeros(len(known_drugs), 2)
#     # all_y_true = []
#     # all_y_pred = []
#     # all_y_prob = []
#
#     # for fold, (train_ids, test_ids) in enumerate(kfold.split(jak_df)):
#     #
#     #     test_df = jak_df.iloc[test_ids].reset_index(drop=True)
#     #     print(test_df)
#     #     test_data = jak_dataset(test_df)
#     #     test_loader = DataLoader(test_data, **params)
#     #     train_df = jak_df.iloc[train_ids].reset_index(drop=True)
#
#     train_df, test_df = train_test_split(jak_df, test_size=0.2, random_state=42)
#     # train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
#     train_df = train_df.reset_index(drop=True)
#     test_df = test_df.reset_index(drop=True)
#     # val_df = val_df.reset_index(drop=True)
#     train_data = jak_dataset(train_df)
#     train_loader = DataLoader(train_data, **params)
#     test_data = jak_dataset(test_df)
#     test_loader = DataLoader(test_data, **params)
#
# ##使用了 CNNforclassification 作为模型，加载了基于 encoder 的预训练模型
#     c = CNNforclassification(max_len, len(vocabulary)).cuda()
#
#     optimizer = optim.SGD(params=c.parameters(), lr=0.1, weight_decay=1e-2)
#     loss_function = nn.CrossEntropyLoss(weight=weight_dict[jak_number].cuda())
#     c.train()
#     for epoch in range(epoches):
#         print("EPOCH -- {}".format(epoch))
#         total_loss = 0
#         for idx, (x, y_true) in tqdm(enumerate(train_loader), total=len(train_loader)):
#             # print(y_true)
#             optimizer.zero_grad()
#             if cuda_available:
#                 x=x.cuda()
#                 y_true=y_true.cuda()
#             output = c(x)
#             loss = loss_function(output, y_true)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#     torch.save(c.state_dict(),
#                'checkpoints/cnn19_for_classification_jak_' + str(jak_number) + ".pt")
#
#
#     c.eval()
#     accumulate_y_pred = []
#     accumulate_y_true = []
#     accumulate_y_prob = []
#     for idx, (X, y_true) in tqdm(enumerate(test_loader), total=len(test_loader)):
#         output = c(X.cuda())
#         _, y_pred = torch.max(output, 1)
#         accumulate_y_pred.extend(y_pred.tolist())
#         accumulate_y_true.extend(y_true.tolist())
#         accumulate_y_prob.extend(torch.softmax(output,1)[:,1].tolist())
#
#     print(
#         classification_report(accumulate_y_true, accumulate_y_pred, target_names=['active', 'inactive'], digits=3))
#        # 1. 混淆矩阵可视化
#     cm = confusion_matrix(accumulate_y_true, accumulate_y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['inactive', 'active'], yticklabels=['inactive', 'active'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()
#
# # 2. 绘制 ROC 曲线和 AUC
#     fpr, tpr, thresholds = roc_curve(accumulate_y_true, accumulate_y_prob)
#     roc_auc = auc(fpr, tpr)
#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     plt.show()
#
# # 3. 绘制 Precision-Recall 曲线
#     precision, recall, _ = precision_recall_curve(accumulate_y_true, accumulate_y_prob)
#     plt.figure(figsize=(6, 5))
#     plt.plot(recall, precision, color='blue', lw=2)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.show()
#
# # 4. 绘制分类指标的条形图
#     f1 = f1_score(accumulate_y_true, accumulate_y_pred)
#     precision = precision_score(accumulate_y_true, accumulate_y_pred)
#     recall = recall_score(accumulate_y_true, accumulate_y_pred)
#
#     metrics = {'Precision': precision, 'Recall': recall, 'F1 Score': f1}
#     metric_names = list(metrics.keys())
#     metric_values = list(metrics.values())
#
#     plt.figure(figsize=(8, 6))
#     plt.bar(metric_names, metric_values, color='skyblue')
#     plt.xlabel('Metrics')
#     plt.ylabel('Scores')
#     plt.title('Classification Metrics')
#     plt.ylim([0, 1])
#     plt.show()
#
#     for idx, (X, y_true) in tqdm(enumerate(known_loader), total=len(known_loader)):
#         output = c(X.cuda())
#         _, y_pred = torch.max(output, 1)
#         print(torch.max(torch.softmax(output, 1), 1)[0].tolist())
#         print(y_pred.tolist())
#
#     for idx, bit in enumerate(y_pred.tolist()):
#         acc_know_pred[idx][bit] += 1
#
#     print(torch.max(acc_know_pred, 1)[1])
#     tn, fp, fn, tp = confusion_matrix(accumulate_y_true, accumulate_y_pred).ravel()
#     print("accuracy:", (tp + tn) / (tn + fp + fn + tp))
#     print("weighted accuracy:", (tp / (fn + tp) + tn / (tn + fp)) / 2)
#     print("precision:", tp / (fp + tp))
#     print("recall:", tp / (fn + tp))
#     print("neg recall:", tn / (tn + fp))
#     print("F1:", tp / (tp + 0.5 * (fp + fn)))
#     print("AUC", roc_auc_score(accumulate_y_true, accumulate_y_prob, average='macro'))
#     print("MCC:", (tp * tn - fn * fp) / math.sqrt((tp + fp) * (tp + fn) * (tn + fn) * (tn + fp)))
#     print("AP:", average_precision_score(accumulate_y_true, accumulate_y_prob, average='macro', pos_label=1))
#
#
#
#
import pickle
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from itertools import cycle  # 用于循环颜色和线型
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 数据预处理函数（保持不变）
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    """将 SMILES 字符串转换为分子指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return fingerprint

# 在读取数据后进行检查（保持不变）
def filter_invalid_smiles(df):
    valid_indices = []
    for i, smiles in enumerate(df['canonical_smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_indices.append(i)
    return df.iloc[valid_indices].reset_index(drop=True)

# 数据集类（保持不变）
class jak_dataset(Dataset):
    def __init__(self, dataframe):
        super(jak_dataset, self).__init__()
        self.len = len(dataframe)
        self.dataframe = dataframe

    def __getitem__(self, idx):
        sml = self.dataframe.canonical_smiles[idx]
        y = 1 if self.dataframe.Activity[idx] == 1 else 0
        return sml, y

    def __len__(self):
        return self.len

# 定义 CNN 模型（修改设备相关参数，默认使用CPU）
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

# 封装 CNN 模型为 scikit-learn 兼容的类（重点修改设备部分）
class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=1e-5, batch_size=16, weight_decay=1e-2, epochs=20, device='cpu',  # 默认改为'cpu'
                 input_size=1, hidden_size=16, output_size=2, pretrained_model_path=None):
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device  # 使用CPU设备
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained_model_path = pretrained_model_path
        
        # 初始化模型并移动到CPU
        self.model = CNNClassifier(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_function = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # 加载预训练模型（如果有的话，确保模型在CPU上加载）
        if pretrained_model_path:
            pretrained_dict = torch.load(pretrained_model_path, map_location=device)  # 指定map_location为CPU
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def fit(self, X, y):
        dataset = jak_dataset(pd.DataFrame({'canonical_smiles': X, 'Activity': y}))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for idx, (x, y_true) in enumerate(loader):
                valid_fingerprints = []
                valid_y_true = []
                
                # 处理SMILES到指纹的转换
                for smiles, label in zip(x, y_true):
                    fingerprint = smiles_to_fingerprint(smiles)
                    if fingerprint is not None:
                        valid_fingerprints.append(torch.tensor(fingerprint, dtype=torch.float32))
                        valid_y_true.append(label)
                
                if len(valid_fingerprints) == 0:
                    continue
                
                # 转换为CPU张量
                x = torch.stack(valid_fingerprints).unsqueeze(1).to(self.device)
                y_true = torch.tensor(valid_y_true, device=self.device)  # 显式指定设备
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_function(output, y_true)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
            self.scheduler.step(avg_loss)
        return self

    def predict(self, X):
        dataset = jak_dataset(pd.DataFrame({'canonical_smiles': X, 'Activity': [0] * len(X)}))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        y_pred = []
        
        with torch.no_grad():
            for idx, (x, _) in enumerate(loader):
                fingerprints = []
                for smiles in x:
                    fingerprint = smiles_to_fingerprint(smiles)
                    if fingerprint is not None:
                        fingerprints.append(torch.tensor(fingerprint, dtype=torch.float32))
                
                if len(fingerprints) == 0:
                    continue
                
                # 转换为CPU张量
                x = torch.stack(fingerprints).unsqueeze(1).to(self.device)
                output = self.model(x)
                _, preds = torch.max(output, 1)
                y_pred.extend(preds.tolist())
        return np.array(y_pred)

    def predict_proba(self, X):
        dataset = jak_dataset(pd.DataFrame({'canonical_smiles': X, 'Activity': [0] * len(X)}))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        y_prob = []
        
        with torch.no_grad():
            for idx, (x, _) in enumerate(loader):
                fingerprints = []
                for smiles in x:
                    fingerprint = smiles_to_fingerprint(smiles)
                    if fingerprint is not None:
                        fingerprints.append(torch.tensor(fingerprint, dtype=torch.float32))
                
                if len(fingerprints) == 0:
                    continue
                
                # 转换为CPU张量
                x = torch.stack(fingerprints).unsqueeze(1).to(self.device)
                output = self.model(x)
                prob = torch.softmax(output, 1)[:, 1]
                y_prob.extend(prob.tolist())
        return np.array(y_prob)

    # 保持get_params和set_params不变
    def get_params(self, deep=True):
        return {
            'lr': self.lr,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'device': self.device,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'pretrained_model_path': self.pretrained_model_path
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# 自定义 AUC 评分函数（保持不变）
def custom_auc(y_true, y_pred, **kwargs):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0  # 如果无法计算 AUC，返回默认值

scorer = make_scorer(custom_auc, needs_proba=True)

if __name__ == '__main__':
    # 文件路径（保持不变）
    file_paths = [
        r'D:\CC\PycharmProjects\GoGT\JAK_ML\data\unique_Homo_um_pact_actJAK1.xlsx',
        r'D:\CC\PycharmProjects\GoGT\JAK_ML\data\unique_Homo_um_pact_actJAK2.xlsx',
        r'D:\CC\PycharmProjects\GoGT\JAK_ML\data\unique_Homo_um_pact_actJAK3.xlsx',
        r'D:\CC\PycharmProjects\GoGT\JAK_ML\data\unique_Homo_um_pact_actTYK2.xlsx'
    ]
    dataset_names = ['JAK1', 'JAK2', 'JAK3', 'TYK2']  # 简洁的数据集名称，与你要求的命名对应

    # 超参数网格（保持不变）
    param_grid = {
        'lr': [1e-5, 5e-5],
        'batch_size': [16],
        'weight_decay': [1e-2],
        'epochs': [10]
    }

    # 强制使用CPU设备
    device = torch.device("cpu")
    
    # 预训练模型路径（如果模型是在CPU上训练的，路径保持不变；如果是GPU训练的，可能需要重新训练）
    pretrained_model_path = 'D:\CC\PycharmProjects\GoGT\checkpoints\cnn_pretrain_27.pt'  # 请确保该模型是CPU版本或已正确转换

    all_best_models = []
    all_metrics = []
    all_roc_data = {}  # 存储所有ROC数据，结构：{dataset_name: {model_name: {'fpr':..., 'tpr':..., 'roc_auc':...}}}
    model_name = "CNN"  # 模型名称，用于标题和存储

    # 初始化all_roc_data结构
    for ds_name in dataset_names:
        if ds_name not in all_roc_data:
            all_roc_data[ds_name] = {}
        all_roc_data[ds_name][model_name] = {}

    plt.figure(figsize=(8, 6))  # 绘图基础尺寸

    # 定义颜色和线型循环器
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])

    for i, file_path in enumerate(file_paths):
        jak_df = pd.read_excel(file_path)
        
        # 数据清洗（保持不变）
        if jak_df.isna().any().any():
            print(f"Data in {file_path} contains NaN values. Removing them...")
            jak_df = jak_df.dropna(subset=['canonical_smiles', 'Activity'])
        jak_df = filter_invalid_smiles(jak_df)
        
        # 划分数据集（保持不变）
        train_df, test_df = train_test_split(jak_df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        X_train = train_df['canonical_smiles'].values
        y_train = train_df['Activity'].values
        X_val = val_df['canonical_smiles'].values
        y_val = val_df['Activity'].values
        X_test = test_df['canonical_smiles'].values
        y_test = test_df['Activity'].values

        # 初始化模型时指定device='cpu'
        model = CNNWrapper(device=device, pretrained_model_path=pretrained_model_path)
        
        # GridSearchCV（保持不变）
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            verbose=3,
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Dataset corresponding to {file_path} Best parameters found: ", grid_search.best_params_)
        best_model = grid_search.best_estimator_
        all_best_models.append(best_model)

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Dataset corresponding to {file_path} Metrics:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print()

        all_metrics.append([accuracy, precision, recall, f1, roc_auc])
        
        # 计算ROC数据并存储
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        all_roc_data[dataset_names[i]][model_name]['fpr'] = fpr
        all_roc_data[dataset_names[i]][model_name]['tpr'] = tpr
        all_roc_data[dataset_names[i]][model_name]['roc_auc'] = roc_auc

    # 单独绘制CNN模型在各数据集上的ROC曲线，按照你要求的样式
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # 随机猜测线

    for ds_name in dataset_names:
        roc_data = all_roc_data[ds_name].get(model_name, {})
        if not roc_data or 'fpr' not in roc_data or 'tpr' not in roc_data:
            print(f"No valid ROC data for {model_name} on {ds_name}")
            continue
        
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        roc_auc_val = roc_data['roc_auc']
        
        plt.plot(fpr, tpr, 
                 color=next(colors),
                 linestyle=next(linestyles),
                 linewidth=2,
                 label=f'{ds_name} (AUC = {roc_auc_val:.2f})')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {model_name} on Model', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图片
    os.makedirs('./roc_curves', exist_ok=True)
    plt.savefig(f'./roc_curves/{model_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 指标汇总和模型保存
    print("综合指标汇总：")
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    for i, metrics in enumerate(all_metrics):
        print(f"Dataset {dataset_names[i]}:")
        for j, metric_name in enumerate(metrics_names):
            print(f"{metric_name}: {metrics[j]:.3f}")
        print()

    # 保存模型
    for i, (best_model, ds_name) in enumerate(zip(all_best_models, dataset_names)):
        with open(f'new_best_cnn_model_dataset_{ds_name}_cpu.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model for {ds_name} saved to: new_best_cnn_model_dataset_{ds_name}_cpu.pkl")