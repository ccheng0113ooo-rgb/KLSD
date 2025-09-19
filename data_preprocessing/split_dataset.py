# # ##=== Final Evaluation ===
# # ##Best Validation R2: 0.5911
# # ##Last Val R2: 0.5823
# ##   boxcox
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import QuantileTransformer, RobustScaler
# import logging
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import os
# import joblib
# from scipy.stats import ks_2samp
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class DataPreprocessor:
#     def __init__(self):
#         self.scaler = None
#         self.boxcox_lambda = None  # 新增：存储Box-Cox参数
#     def plot_activity_distribution(self, df, title):
#         """可视化活性值分布"""
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         sns.histplot(df['activity'], kde=True)
#         plt.title(f'{title} Raw Activity')
        
#         plt.subplot(1, 2, 2)
#         sns.histplot(df['value_transformed'], kde=True)
#         plt.title(f'{title} Transformed Activity')
#         plt.tight_layout()
#         plt.savefig(f"{title.replace(' ', '_')}_activity_dist_tyk2.png")
#         plt.close()

#     def canonicalize_smiles(self, smiles):
#         """标准化SMILES并验证有效性"""
#         try:
#             mol = Chem.MolFromSmiles(str(smiles))
#             if mol:
#                 Chem.SanitizeMol(mol)
#                 return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
#             return None
#         except Exception as e:
#             logger.warning(f"SMILES标准化失败: {smiles} | 错误: {str(e)}")
#             return None

#     def _validate_processed_data(self, train_df, val_df, test_df):
#         """增强的数据验证"""
#         logger.info("开始验证处理后数据...")
    
#         # 检查标准化值范围
#         for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
#             values = df['value_transformed']
#             assert values.between(-8, 8).all(), f"{name}值范围异常: {values.min():.2f}~{values.max():.2f}"
        
#             # 检查SMILES有效性
#             invalid = df['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x) is None).sum()
#             assert invalid == 0, f"{name}发现 {invalid} 个无效SMILES"

#         # KS检验分布一致性（两两比较）
#         datasets = [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]
#         for i in range(len(datasets)):
#             for j in range(i+1, len(datasets)):
#                 name1, df1 = datasets[i]
#                 name2, df2 = datasets[j]
#                 ks_stat, p_val = ks_2samp(df1['value_transformed'], df2['value_transformed'])
#                 logger.info(f"{name1} vs {name2} KS检验 p值: {p_val:.4f}")
#                 assert p_val > 0.05, f"{name1}/{name2}分布不一致 (p={p_val:.3f})"

#         # 检查数据泄漏
#         combined = pd.concat([train_df, val_df, test_df])
#         duplicates = combined.duplicated(subset=['canonical_smiles'], keep=False)
#         assert duplicates.sum() == 0, f"发现 {duplicates.sum()} 个重复SMILES"

#         logger.info("✓ 处理后数据验证通过")

#     def _stratified_split(self, df, val_size=0.2, test_size=0.2):
#         """分层划分Train/Val/Test"""
#         # 首次划分 (train+val) vs test
#         train_val, test = train_test_split(
#             df,
#             test_size=test_size,
#             random_state=42,
#             stratify=pd.qcut(df['value_transformed'], q=5)
#         )
        
#         # 二次划分 train vs val
#         relative_val_size = val_size / (1 - test_size)
#         train, val = train_test_split(
#             train_val,
#             test_size=relative_val_size,
#             random_state=42,
#             stratify=pd.qcut(train_val['value_transformed'], q=5)
#         )
        
#         return train, val, test

#     def advanced_transform(self, values, method='robust'):
#         """改进的稳健标准化，增加Box-Cox和Quantile选项"""
#         values = np.array(values)

#         if method == 'boxcox':
#             # Box-Cox变换要求正值
#             shifted = values - values.min() + 1e-6
#             transformed, self.boxcox_lambda = stats.boxcox(shifted)
#             self.scaler = RobustScaler(quantile_range=(5, 95))
#             transformed = self.scaler.fit_transform(transformed.reshape(-1, 1)).flatten()
#         elif method == 'quantile':
#             self.scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
#             return self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
#         else:  # 默认使用robust
#             # 阶段1：log1p变换
#             transformed = np.log1p(values - values.min() + 1e-6)
#             # 阶段2：RobustScaler
#             self.scaler = RobustScaler(quantile_range=(5, 95))
#             transformed = self.scaler.fit_transform(transformed.reshape(-1, 1)).flatten()

#         return transformed
#     # def advanced_transform(self, values):
#     #     """改进的稳健标准化"""
#     #     values = np.array(values)
        
#     #     # 阶段1：log1p变换
#     #     transformed = np.log1p(values - values.min() + 1e-6)
        
#     #     # 阶段2：RobustScaler
#     #     self.scaler = RobustScaler(quantile_range=(5, 95))
#     #     transformed = self.scaler.fit_transform(transformed.reshape(-1, 1)).flatten()
        
#     #     return transformed

#     def calculate_optimal_length(self, smiles_list, coverage=0.95):
#         """动态计算最佳序列长度"""
#         lengths = []
#         for smi in smiles_list:
#             canon_smi = self.canonicalize_smiles(smi)
#             if canon_smi:
#                 lengths.append(len(canon_smi))
        
#         if not lengths:
#             raise ValueError("无法计算长度：无有效SMILES")
            
#         optimal_len = int(np.percentile(lengths, coverage * 100)) + 2
#         logger.info(f"计算出的最优序列长度: {optimal_len} (覆盖{coverage*100:.0f}%样本)")
        
#         # 可视化长度分布
#         plt.figure(figsize=(10, 6))
#         plt.hist(lengths, bins=30, edgecolor='black')
#         plt.axvline(optimal_len, color='red', linestyle='--', label=f'Optimal Length: {optimal_len}')
#         plt.title('SMILES Length Distribution')
#         plt.xlabel('Length')
#         plt.ylabel('Count')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig("smiles_length_distribution_tyk2.png")
#         plt.close()
        
#         return optimal_len

#     def process_data(self, input_csv, output_dir, test_size=0.2, val_size=0.2, transform_method='robust'):
#         """完整数据处理流程"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 1. 加载并清洗数据
#         df = pd.read_csv(input_csv)
#         if 'SMILES' not in df.columns:
#             raise ValueError("输入CSV必须包含'SMILES'列")
            
#         logger.info(f"原始数据量: {len(df)}")
#         df['canonical_smiles'] = df['SMILES'].apply(self.canonicalize_smiles)
#         valid_df = df[df['canonical_smiles'].notna()].copy()
#         logger.info(f"有效SMILES数据量: {len(valid_df)}")
        
#         # 2. 活性值处理
#         valid_df = valid_df[(valid_df['activity'] > 0) & (valid_df['activity'] < 1e6)]
#         log_values = np.log1p(valid_df['activity'])
#         q1, q3 = np.percentile(log_values, [25, 75])
#         iqr = q3 - q1
#         bounds = np.exp([q1 - 2.5*iqr, q3 + 2.5*iqr]) - 1
#         filtered_df = valid_df[(valid_df['activity'] >= bounds[0]) & 
#                              (valid_df['activity'] <= bounds[1])]
        
#         # 去除重复的canonical_smiles
#         filtered_df = filtered_df.drop_duplicates(subset=['canonical_smiles'], keep='first')
#         logger.info(f"去重后数据量: {len(filtered_df)}")
        
#         # 3. 数据转换（新增transform_method参数）
#         transformed = self.advanced_transform(filtered_df['activity'].values, method=transform_method)
#         filtered_df['value_transformed'] = transformed
        
#         # 可视化活性值分布
#         self.plot_activity_distribution(filtered_df, "Processed")
        
#         # 4. 三层划分 (60/20/20)
#         train_df, val_df, test_df = self._stratified_split(filtered_df, val_size, test_size)
        
#         # 5. 验证数据
#         self._validate_processed_data(train_df, val_df, test_df)

#         # 6. 保存数据（新增保存boxcox参数）
#         joblib.dump({
#             'scaler': self.scaler,
#             'boxcox_lambda': self.boxcox_lambda if transform_method == 'boxcox' else None,
#             'train_min_activity': filtered_df['activity'].min(),  # 保存训练集活性最小值
#             'transform_method': transform_method,
#             'epsilon': 1e-6  # 保存偏移量参数，避免硬编码
#         }, os.path.join(output_dir, 'transformers_tyk2.joblib'))
        
#         train_df.to_csv(os.path.join(output_dir, 'train_processed_tyk2.csv'), index=False)
#         val_df.to_csv(os.path.join(output_dir, 'val_processed_tyk2.csv'), index=False)
#         test_df.to_csv(os.path.join(output_dir, 'test_processed_tyk2.csv'), index=False)
        
#         # 7. 计算序列长度 (使用全量数据)
#         optimal_len = self.calculate_optimal_length(
#             pd.concat([train_df, val_df, test_df])['canonical_smiles'].tolist()
#         )
        
#         return {
#             'train_size': len(train_df),
#             'val_size': len(val_df),
#             'test_size': len(test_df),
#             'optimal_seq_length': optimal_len,
#             'transform_method': transform_method
#         }

# if __name__ == "__main__":
#     processor = DataPreprocessor()
#     result = processor.process_data(
#         input_csv=r"D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\valid_activity_tyk2.csv",
#         output_dir=r"D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed",
#         test_size=0.2,
#         val_size=0.2,
#         transform_method='boxcox'  # 可选项：'robust' 或 'boxcox'(效果最好)或 'quantile'
#     )
#     logger.info(f"处理完成！数据集划分: Train={result['train_size']}, Val={result['val_size']}, Test={result['test_size']}")
#     logger.info(f"最优序列长度: {result['optimal_seq_length']}")
#     logger.info(f"使用的变换方法: {result['transform_method']}")





##将参数封装为字典后再保存
# ##=== Final Evaluation ===
# ##Best Validation R2: 0.5911
# ##Last Val R2: 0.5823
##   boxcox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import joblib
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.boxcox_lambda = None  # 新增：存储Box-Cox参数
        self.train_min_activity = None  # 新增：存储训练集最小活性值
        self.transform_method = None  # 新增：存储变换方法
        self.epsilon = 1e-6  # 新增：存储偏移量
    
    def plot_activity_distribution(self, df, title):
        """可视化活性值分布"""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df['activity'], kde=True)
        plt.title(f'{title} Raw Activity')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['value_transformed'], kde=True)
        plt.title(f'{title} Transformed Activity')
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}_activity_dist_tyk2.png")
        plt.close()

    def canonicalize_smiles(self, smiles):
        """标准化SMILES并验证有效性"""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return None
        except Exception as e:
            logger.warning(f"SMILES标准化失败: {smiles} | 错误: {str(e)}")
            return None

    def _validate_processed_data(self, train_df, val_df, test_df):
        """增强的数据验证"""
        logger.info("开始验证处理后数据...")
    
        # 检查标准化值范围
        for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
            values = df['value_transformed']
            assert values.between(-8, 8).all(), f"{name}值范围异常: {values.min():.2f}~{values.max():.2f}"
        
            # 检查SMILES有效性
            invalid = df['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x) is None).sum()
            assert invalid == 0, f"{name}发现 {invalid} 个无效SMILES"

        # KS检验分布一致性（两两比较）
        datasets = [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                name1, df1 = datasets[i]
                name2, df2 = datasets[j]
                ks_stat, p_val = ks_2samp(df1['value_transformed'], df2['value_transformed'])
                logger.info(f"{name1} vs {name2} KS检验 p值: {p_val:.4f}")
                assert p_val > 0.05, f"{name1}/{name2}分布不一致 (p={p_val:.3f})"

        # 检查数据泄漏
        combined = pd.concat([train_df, val_df, test_df])
        duplicates = combined.duplicated(subset=['canonical_smiles'], keep=False)
        assert duplicates.sum() == 0, f"发现 {duplicates.sum()} 个重复SMILES"

        logger.info("✓ 处理后数据验证通过")

    def _stratified_split(self, df, val_size=0.2, test_size=0.2):
        """分层划分Train/Val/Test"""
        # 首次划分 (train+val) vs test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=pd.qcut(df['value_transformed'], q=5)
        )
        
        # 二次划分 train vs val
        relative_val_size = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=42,
            stratify=pd.qcut(train_val['value_transformed'], q=5)
        )
        
        return train, val, test

    def advanced_transform(self, values, method='robust'):
        """改进的稳健标准化，增加Box-Cox和Quantile选项"""
        values = np.array(values)
        self.transform_method = method  # 保存变换方法
        self.train_min_activity = values.min()  # 保存训练集最小活性值
        
        if method == 'boxcox':
            # Box-Cox变换要求正值
            shifted = values - self.train_min_activity + self.epsilon
            transformed, self.boxcox_lambda = stats.boxcox(shifted)
            self.scaler = RobustScaler(quantile_range=(5, 95))
            transformed = self.scaler.fit_transform(transformed.reshape(-1, 1)).flatten()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
            return self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        else:  # 默认使用robust
            # 阶段1：log1p变换
            transformed = np.log1p(values - self.train_min_activity + self.epsilon)
            # 阶段2：RobustScaler
            self.scaler = RobustScaler(quantile_range=(5, 95))
            transformed = self.scaler.fit_transform(transformed.reshape(-1, 1)).flatten()

        return transformed

    def calculate_optimal_length(self, smiles_list, coverage=0.95):
        """动态计算最佳序列长度"""
        lengths = []
        for smi in smiles_list:
            canon_smi = self.canonicalize_smiles(smi)
            if canon_smi:
                lengths.append(len(canon_smi))
        
        if not lengths:
            raise ValueError("无法计算长度：无有效SMILES")
            
        optimal_len = int(np.percentile(lengths, coverage * 100)) + 2
        logger.info(f"计算出的最优序列长度: {optimal_len} (覆盖{coverage*100:.0f}%样本)")
        
        # 可视化长度分布
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, edgecolor='black')
        plt.axvline(optimal_len, color='red', linestyle='--', label=f'Optimal Length: {optimal_len}')
        plt.title('SMILES Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("smiles_length_distribution_tyk2.png")
        plt.close()
        
        return optimal_len

    def process_data(self, input_csv, output_dir, test_size=0.2, val_size=0.2, transform_method='robust'):
        """完整数据处理流程"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 加载并清洗数据
        df = pd.read_csv(input_csv)
        if 'SMILES' not in df.columns:
            raise ValueError("输入CSV必须包含'SMILES'列")
            
        logger.info(f"原始数据量: {len(df)}")
        df['canonical_smiles'] = df['SMILES'].apply(self.canonicalize_smiles)
        valid_df = df[df['canonical_smiles'].notna()].copy()
        logger.info(f"有效SMILES数据量: {len(valid_df)}")
        
        # 2. 活性值处理
        valid_df = valid_df[(valid_df['activity'] > 0) & (valid_df['activity'] < 1e6)]
        log_values = np.log1p(valid_df['activity'])
        q1, q3 = np.percentile(log_values, [25, 75])
        iqr = q3 - q1
        bounds = np.exp([q1 - 2.5*iqr, q3 + 2.5*iqr]) - 1
        filtered_df = valid_df[(valid_df['activity'] >= bounds[0]) & 
                             (valid_df['activity'] <= bounds[1])]
        
        # 去除重复的canonical_smiles
        filtered_df = filtered_df.drop_duplicates(subset=['canonical_smiles'], keep='first')
        logger.info(f"去重后数据量: {len(filtered_df)}")
        
        # 3. 数据转换
        transformed = self.advanced_transform(filtered_df['activity'].values, method=transform_method)
        filtered_df['value_transformed'] = transformed
        
        # 可视化活性值分布
        self.plot_activity_distribution(filtered_df, "Processed")
        
        # 4. 三层划分 (60/20/20)
        train_df, val_df, test_df = self._stratified_split(filtered_df, val_size, test_size)
        
        # 5. 验证数据
        self._validate_processed_data(train_df, val_df, test_df)

        # 6. 保存数据和完整的transformers参数
        transformers = {
            'scaler': self.scaler,
            'boxcox_lambda': self.boxcox_lambda,
            'train_min_activity': self.train_min_activity,
            'transform_method': self.transform_method,
            'epsilon': self.epsilon,
            'train_mean_activity': np.mean(filtered_df['activity'].values),  # 新增：训练集平均活性值
            'train_std_activity': np.std(filtered_df['activity'].values),    # 新增：训练集活性值标准差
        }
        
        joblib.dump(transformers, os.path.join(output_dir, 'transformers_tyk2.joblib'))
        
        train_df.to_csv(os.path.join(output_dir, 'train_processed_tyk2.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_processed_tyk2.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_processed_tyk2.csv'), index=False)
        
        # 7. 计算序列长度 (使用全量数据)
        optimal_len = self.calculate_optimal_length(
            pd.concat([train_df, val_df, test_df])['canonical_smiles'].tolist()
        )
        
        return {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'optimal_seq_length': optimal_len,
            'transform_method': transform_method
        }

if __name__ == "__main__":
    processor = DataPreprocessor()
    result = processor.process_data(
        input_csv=r"D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\valid_activity_tyk2.csv",
        output_dir=r"D:\desktop\JAKInhibition-master\JAKInhibition-master\finaldata\processed",
        test_size=0.2,
        val_size=0.2,
        transform_method='boxcox'  # 可选项：'robust' 或 'boxcox'(效果最好)或 'quantile'
    )
    logger.info(f"处理完成！数据集划分: Train={result['train_size']}, Val={result['val_size']}, Test={result['test_size']}")
    logger.info(f"最优序列长度: {result['optimal_seq_length']}")
    logger.info(f"使用的变换方法: {result['transform_method']}")