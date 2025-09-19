import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import logging
import os
import matplotlib.pyplot as plt
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, input_csv, output_dir="validation_reports"):
        self.input_csv = input_csv
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_checks(self):
        """执行所有验证检查"""
        try:
            # 加载数据
            df = pd.read_csv(self.input_csv)
            logger.info(f"成功加载数据，共 {len(df)} 条记录")
            
            # 自动检测SMILES列名
            smiles_col = self._detect_smiles_column(df)
            if not smiles_col:
                raise ValueError("未找到有效的SMILES列")
                
            # 执行检查
            checks = {
                'missing_values': self.check_missing_values(df),
                'smiles_validity': self.check_smiles_validity(df, smiles_col),
                'activity_distribution': self.check_activity_distribution(df),
                'smiles_length': self.check_smiles_length(df, smiles_col),
                'duplicates': self.check_duplicates(df, smiles_col),
                'fingerprint_diversity': self.check_fingerprint_diversity(df, smiles_col)
            }
            
            # 生成报告
            self.generate_report(checks, smiles_col)
            return True
        except Exception as e:
            logger.error(f"验证过程中出错: {str(e)}", exc_info=True)
            return False

    def _detect_smiles_column(self, df):
        """自动检测SMILES列名"""
        possible_names = ['canonical_smiles', 'SMILES', 'smiles', 'original_smiles']
        for col in possible_names:
            if col in df.columns:
                logger.info(f"检测到SMILES列: {col}")
                return col
        return None

    def check_missing_values(self, df):
        """检查缺失值"""
        result = {'passed': True, 'details': {}}
        missing = df.isnull().sum()
        
        for col, count in missing.items():
            if count > 0:
                result['passed'] = False
                result['details'][col] = f"{count} 个缺失值"
        
        if result['passed']:
            logger.info("缺失值检查: 通过")
        else:
            logger.warning(f"缺失值检查: 失败\n{result['details']}")
        
        return result

    def check_smiles_validity(self, df, smiles_col):
        """验证SMILES有效性"""
        result = {'passed': True, 'valid_count': 0, 'invalid_examples': []}
        
        for i, row in df.iterrows():
            smi = str(row[smiles_col])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                result['passed'] = False
                result['invalid_examples'].append(smi)
                if len(result['invalid_examples']) >= 5:  # 最多记录5个无效例子
                    break
            else:
                result['valid_count'] += 1
        
        validity_rate = result['valid_count'] / len(df)
        logger.info(f"SMILES有效性: {validity_rate:.2%} ({result['valid_count']}/{len(df)})")
        
        if not result['passed']:
            logger.warning(f"无效SMILES示例: {result['invalid_examples'][:3]}...")
        
        return result

    def check_activity_distribution(self, df):
        """检查活性值分布"""
        result = {'passed': True, 'stats': {}}
        
        if 'activity' not in df.columns:
            logger.warning("未找到'activity'列，跳过活性分布检查")
            return {'passed': False, 'error': 'activity列不存在'}
        
        activity = df['activity']
        result['stats'] = {
            'min': activity.min(),
            'max': activity.max(),
            'mean': activity.mean(),
            'std': activity.std(),
            'q1': np.percentile(activity, 25),
            'median': np.median(activity),
            'q3': np.percentile(activity, 75)
        }
        
        # 绘制分布图
        plt.figure(figsize=(10, 6))
        plt.hist(activity, bins=30, edgecolor='black')
        plt.title('Activity Value Distribution')
        plt.xlabel('Activity')
        plt.ylabel('Count')
        plot_path = os.path.join(self.output_dir, 'activity_distribution.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"活性分布图已保存到 {plot_path}")
        
        return result

    def check_smiles_length(self, df, smiles_col):
        """检查SMILES长度分布"""
        lengths = df[smiles_col].apply(len)
        result = {
            'min_len': lengths.min(),
            'max_len': lengths.max(),
            'avg_len': lengths.mean(),
            'length_dist': dict(Counter(lengths))
        }
        
        # 绘制长度分布
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, edgecolor='black')
        plt.title('SMILES Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plot_path = os.path.join(self.output_dir, 'smiles_length_dist.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"SMILES长度: 最小={result['min_len']}, 最大={result['max_len']}, 平均={result['avg_len']:.1f}")
        return result

    def check_duplicates(self, df, smiles_col):
        """检查重复SMILES"""
        duplicates = df.duplicated(subset=[smiles_col], keep=False)
        dup_count = duplicates.sum()
        
        result = {
            'passed': dup_count == 0,
            'duplicate_count': dup_count,
            'examples': df[duplicates][smiles_col].head(3).tolist() if dup_count > 0 else []
        }
        
        if result['passed']:
            logger.info("重复检查: 无重复SMILES")
        else:
            logger.warning(f"重复检查: 发现 {dup_count} 个重复\n示例: {result['examples']}")
        
        return result

    def check_fingerprint_diversity(self, df, smiles_col, sample_size=200):
        """检查分子指纹多样性"""
        try:
            # 采样检查
            sample_df = df.sample(min(sample_size, len(df)))
            fps = []
            valid_smiles = []
            
            for smi in sample_df[smiles_col]:
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
                    valid_smiles.append(smi)
            
            if len(fps) < 2:
                return {'passed': False, 'reason': '有效分子不足(<2)无法计算相似度'}
            
            # 计算相似度矩阵
            from rdkit import DataStructs
            similarity = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    similarity.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            
            result = {
                'passed': True,
                'avg_similarity': np.mean(similarity),
                'min_similarity': np.min(similarity),
                'max_similarity': np.max(similarity),
                'valid_molecules': len(fps)
            }
            
            logger.info(f"指纹多样性: 基于{len(fps)}个有效分子, 平均相似度={result['avg_similarity']:.3f}")
            return result
        except Exception as e:
            logger.warning(f"指纹多样性检查失败: {str(e)}")
            return {'passed': False, 'error': str(e)}

    def generate_report(self, checks, smiles_col):
        """生成HTML格式的验证报告"""
        report_path = os.path.join(self.output_dir, 'validation_report.html')
        
        html = f"""
        <html>
        <head>
            <title>数据验证报告 - {os.path.basename(self.input_csv)}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .check {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .passed {{ background-color: #e8f5e9; }}
                .failed {{ background-color: #ffebee; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>数据验证报告</h1>
            <p><strong>数据集:</strong> {self.input_csv}</p>
            <p><strong>SMILES列:</strong> {smiles_col}</p>
            <p><strong>生成时间:</strong> {pd.Timestamp.now()}</p>
            
            <h2>检查结果摘要</h2>
            <table>
                <tr><th>检查项</th><th>结果</th><th>详情</th></tr>
        """
        
        # 添加检查结果
        for name, check in checks.items():
            status = "通过" if check.get('passed', False) else "失败"
            row_class = "passed" if check.get('passed', False) else "failed"
            
            details = ""
            if 'details' in check:
                details = "<br>".join(f"{k}: {v}" for k, v in check['details'].items())
            elif 'stats' in check:
                details = "<br>".join(f"{k}: {v:.4f}" for k, v in check['stats'].items())
            elif 'error' in check:
                details = check['error']
            
            html += f"""
                <tr class="{row_class}">
                    <td>{name.replace('_', ' ').title()}</td>
                    <td>{status}</td>
                    <td>{details}</td>
                </tr>
            """
        
        # 添加图表
        html += """
            </table>
            
            <h2>数据分布可视化</h2>
            <div style="display: flex; flex-wrap: wrap;">
        """
        
        # 动态添加图表
        plots = ['activity_distribution.png', 'smiles_length_dist.png']
        for plot in plots:
            if os.path.exists(os.path.join(self.output_dir, plot)):
                html += f"""
                <div style="width: 48%; margin: 1%;">
                    <h3>{plot.replace('_', ' ').title().replace('.png', '')}</h3>
                    <img src="{plot}" alt="{plot}">
                </div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"完整验证报告已生成: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='验证化学数据集')
    parser.add_argument('input', help='输入CSV文件路径')
    parser.add_argument('--output', default="validation_reports", help='报告输出目录')
    args = parser.parse_args()
    
    validator = DataValidator(args.input, args.output)
    if validator.run_checks():
        logger.info("数据验证完成")
    else:
        logger.error("数据验证发现重大问题，请检查报告！")