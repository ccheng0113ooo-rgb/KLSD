import numpy.random as npr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 获取颜色映射
my_cmap = plt.get_cmap("viridis")

# 读取数据
df = pd.read_csv(r"D:\desktop\JAKInhibition-master\JAKInhibition-master\data\allme.csv")

# 查看数据集前10行
print(df.head(10))

# 打印整个数据集的统计信息
print("\n Whole dataset statistics \n", df.describe())

# 打印pIC50数据的统计信息
print("\n pIC50 statistics \n", df.loc[df['measurement_type'] == "pIC50"].describe())

# 打印pKi数据的统计信息
print("\n pKi statistics \n", df.loc[df['measurement_type'] == "pKi"].describe())

# 统计唯一分子数量
unique_smiles = df["SMILES"].unique()
number_smiles = len(unique_smiles)
print(f"Dataset contains {number_smiles} unique molecules")

# 统计不同测量类型和激酶的样本数量
labels = []
sample_counts = []
for measurement in ["pIC50", "pKi"]:
    m_df = df.loc[df["measurement_type"] == measurement]
    for enzyme in ["JAK1", "JAK2", "JAK3", "TYK2"]:
        me_df = m_df.loc[m_df["Kinase_name"] == enzyme]
        message = f"\ndataset includes {len(me_df)} {measurement} measurements of {enzyme}"
        print(message)
        labels.append(f"{measurement}:{enzyme}")
        sample_counts.append(len(me_df))

# 绘制样本数量柱状图
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.bar(labels, sample_counts, color=my_cmap(20))
ax.set_ylabel("number of samples")
plt.xticks(rotation=20)
plt.title("Measurement frequency by type and enzyme")
plt.savefig(r"D:\desktop\JAKInhibition-master\JAKInhibition-master\jak\assets\measurement_frequency.png")
plt.show()

# 提取pIC50和pKi的测量值
pic50 = df.loc[df['measurement_type'] == "pIC50"]["measurement_value"].to_numpy()
pki = df.loc[df['measurement_type'] == "pKi"]["measurement_value"].to_numpy()

# 绘制pIC50和pKi的直方图
fig, ax = plt.subplots(1, 1)
ax.hist(pic50[:, None], bins=32, label="pIC50", color=my_cmap(64), alpha=0.25)
ax.set_ylabel("pIC50")
ax2 = ax.twinx()
ax2.hist(pki[:, None], bins=32, label="pKi", color=my_cmap(192), alpha=0.25)
ax2.set_ylabel("pKi")
ax.set_title("pKi and  pIC50")
ax2.legend(loc=1)
ax.legend(loc=2)
plt.tight_layout()
plt.savefig(r"D:\desktop\JAKInhibition-master\JAKInhibition-master\jak\assets\histogram.png")
plt.show()