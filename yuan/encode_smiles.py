##原数据使用的smile编码的代码
# import numpy.random as npr
# import numpy as np
# import torch 
# from jak.transformer import XFormer
# import pandas as pd
# import matplotlib.pyplot as plt

# # 设置颜色映射（虽然在这段代码中未实际使用）
# my_cmap = plt.get_cmap("plasma")

# # 读取数据
# df = pd.read_csv(r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\train_JAK.csv")
# df.head()

# # 定义SMILES词汇表
# smiles_vocab = "#()+-1234567=BCFHINOPS[]cilnors"

# # 初始化模型
# model = XFormer(vocab=smiles_vocab, token_dim=31, seq_length=128, lr=1e-3, device="cpu", tag="inference")

# # 加载预训练模型参数
# model_path = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\parameters\xformer_default_tag_seed42\tag_xformer_default_tag_seed42_epoch99.pt"
# model_state_dict = torch.load(model_path, map_location=model.my_device)
# model.load_state_dict(model_state_dict)

# # 计算特征数量
# num_values = len(df["Kinase_name"].unique()) * len(df["measurement_type"].unique())

# # 初始化空张量和字典
# x = torch.Tensor()
# y = torch.Tensor()
# x_dict = {}
# y_dict = {}

# # 定义测量类型和激酶名称的映射字典
# m_dict = {"pIC50": 0 , "pKi": 1}
# e_dict = {"JAK1": 0, "JAK2": 1, "JAK3": 2, "TYK2": 3}
# number_enzymes = 4

# # 编码SMILES字符串并构建特征和标签字典
# with torch.no_grad():
#     for ii in range(len(df)):
#         my_smile = df["SMILES"][ii]
#         if my_smile not in x_dict.keys():
#             encoded = model.encode(my_smile).reshape(1, -1)
#             x_dict[my_smile] = encoded
#             y_dict[my_smile] = [0.0] * num_values
#         index = m_dict[df["measurement_type"][ii]] * number_enzymes + e_dict[df["Kinase_name"][ii]]
#         y_dict[my_smile][index] = df["measurement_value"][ii]

# # 合并特征和标签张量
# x = torch.Tensor()
# y = torch.Tensor()
# for key in x_dict.keys():
#     x = torch.cat([x, x_dict[key]])
#     y = torch.cat([y, torch.tensor(y_dict[key]).reshape(1, num_values)])

# # 打印特征和标签的形状
# print(x.shape, y.shape)

# # 保存特征和标签
# x_filepath = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\jakinase_x"
# y_filepath = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\jakinase_y"
# torch.save(x, f"{x_filepath}.pt")
# torch.save(y, f"{y_filepath}.pt")
# np.save(f"{x_filepath}.npy", x.numpy())
# np.save(f"{y_filepath}.npy", y.numpy())

# rdkit_featurizer.py
##为训练集和测试集生成RDKit分子特征
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_features(smiles_path, output_path):
    df = pd.read_csv(smiles_path)
    features = []
    invalid_count = 0
    
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            features.append([Descriptors.MolWt(mol), *fp])
        else:
            invalid_count += 1
    
    features = np.array(features)
    np.save(output_path, features)
    logger.info(f"生成文件: {output_path} | 有效样本: {len(features)} | 无效SMILES: {invalid_count}")

if __name__ == "__main__":
    # 训练集特征
    generate_features(
        r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\train_JAK_18086.csv",
        r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\rdkit_features_train_18086.npy"
    )
    
    # 测试集特征
    generate_features(
        r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\test_JAK_4522.csv",
        r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\rdkit_features_test_4522.npy"
    )