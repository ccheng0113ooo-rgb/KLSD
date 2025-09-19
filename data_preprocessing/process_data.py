##=== Final Evaluation ===
##Best Validation R2: 0.5911
##Last Val R2: 0.5823
"""
optimized_processor.py
修复氢原子显示和列名问题的增强版处理器
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def visualize_smiles_length_distribution(smiles_list, title):
    """可视化SMILES长度分布"""
    lengths = [len(smi) for smi in smiles_list]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black')
    plt.title(f'{title} SMILES Length Distribution')
    plt.xlabel('SMILES Length')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{title.replace(' ', '_')}_length_dist_tyk2.png")
    plt.close()

def process_data(input_path, output_dir="four_data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据并过滤标题行
    df = pd.read_csv(input_path)
    if 'SMILES' in df.columns:
        df = df[df['SMILES'] != 'SMILES']
    
    # 处理数据
    results = []
    invalid_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
        smi = str(row.iloc[0]).strip()  # 第一列为SMILES
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # 修改点1：移除Kekulization，保留芳香性
                canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                results.append({
                    'original_smiles': smi,
                    'canonical_smiles': canon_smi,
                    'activity': round(float(row.iloc[1]), 2)  # 第二列为活性值
                })
            else:
                invalid_count += 1
        except Exception as e:
            invalid_count += 1
            continue
    
    # 生成DataFrame
    valid_df = pd.DataFrame(results)
    
    # 可视化原始与规范化SMILES长度对比
    if len(results) > 0:
        visualize_smiles_length_distribution(
            valid_df['original_smiles'].tolist(), 
            "Original"
        )
        visualize_smiles_length_distribution(
            valid_df['canonical_smiles'].tolist(), 
            "Canonical"
        )
    
    # 保存结果
    valid_df.to_csv(f"{output_dir}/valid_activity_tyk2.csv", index=False)
    
    # 生成详细报告
    with open(f"{output_dir}/processing_tyk2_report.txt", 'w') as f:
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Valid Records: {len(valid_df)}\n")
        f.write(f"Invalid Records: {invalid_count}\n")
        f.write(f"Unchanged SMILES: {sum(valid_df['original_smiles'] == valid_df['canonical_smiles'])}\n")
        f.write("\nSample Output:\n")
        f.write(valid_df.head(10).to_string(index=False))
    
    return valid_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("--output", default="clean_littledata", help="Output directory")
    args = parser.parse_args()
    
    print("="*50)
    print(f"Processing: {args.input}")
    result = process_data(args.input, args.output)
    print("\nSample Output:")
    print(result.head(3).to_string(index=False))
    print("="*50)
    print(f"Saved results to {args.output} directory")




