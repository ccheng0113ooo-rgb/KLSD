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