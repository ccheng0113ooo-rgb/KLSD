# from function import *
# # from jak_dataset import *
# from sklearn.neighbors import KNeighborsClassifier
# from ast import increment_lineno
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
# import pickle
# import os


# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# header = ['bit'+str(i) for i in range(167)]
# import os
# def create_path(path):
#     isExist = os.path.exists(path)
#     print(path, ' folder is in directory: ', isExist)
#     if not isExist:
#         os.makedirs(path)
#         print(path, " is created!")

# model_path = "model/"
# create_path(model_path)

# def KNN_rough_tune(data, enzyme): 
#     # global header
#     header = ['bit'+str(i) for i in range(167)]
#     X = data[header]
#     y = data['Activity']
#     # k_range = range(1, 50)
#     # k_list = [item * 10 for item in k_range]
#     # print('First rough test neighbors 1-500')
#     # 动态确定最大允许的 n_neighbors
#     max_neighbors = len(X) - 1  # 样本数减去 1，避免包含自己作为邻居
    
#     if max_neighbors < 50:
#         print(f"Warning: {enzyme} 数据样本非常少，仅 {len(X)} 条记录，邻居数将动态调整。")
    
#     # 确保k_range的最大值不会超过样本数
#     k_range = range(1, min(500, max_neighbors) + 1)  # 取较小值，保证不超过最大样本数
#     k_list = [item for item in k_range if item <= max_neighbors]  # 确保k_list不超过最大邻居数

#     print(f"Rough test neighbors for {enzyme} with {len(X)} samples (max neighbors = {max_neighbors}):")
#     print(f"Testing k values: {k_list}")


#     k_scores = []
#     # Cross validation 5
#     for k in k_list:
#         if k%50 == 0:
#             print('Now trying neighbor#', k)
#         knn = KNeighborsClassifier(n_neighbors=k)
#         scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#         k_scores.append(scores.mean())

#     plt.plot(k_list, k_scores)
#     plt.xlabel('Value of neighbor k for KNN')
#     plt.ylabel('5-fold cross-validated accuracy')
#     title = enzyme + ' KNN accuracy with neighbors ' + 'rough tune'
#     plt.title(title)
#     path = "knn_figures/"
#     # Check whether the specified path exists or not
#     isExist = os.path.exists(path)
#     #printing if the path exists or not
#     print(path, ' folder is in directory: ', isExist)
#     if not isExist:
#     # Create a new directory because it does not exist
#         os.makedirs(path)
#         print(path, " is created!")
#     plt.savefig(path + enzyme + '_rough_tune.png')
#     plt.close()
#     print('Rough tune max k_scores: ')
#     print(max(k_scores))
#     max_accuracy = max(k_scores)
#     neighbors_rough = []
#     for i, j in zip(k_list, k_scores):
#         if j == max_accuracy:
#             print('Rough tune max_accuracy occurs at neighbor #', i)   
#             neighbors_rough.append(i)
            
#             # Further fine tune

#     print('neighbors_rough for ', enzyme, ' : ', neighbors_rough)
#     print(' max accuracy: ', max_accuracy)
#     return neighbors_rough[0]

# def KNN_fine_tune(data, enzyme='JAK1', rough_neighbor=10):
#     header = ['bit'+str(i) for i in range(167)]
#     X = data[header]
#     y = data['Activity']
#     low_bound = max(rough_neighbor - 10, 1)
#     up_bound = low_bound + 20 
#     k_range = range(low_bound, up_bound)
#     k_scores = []
#     for k in k_range:
#         if k%5 == 0:
#             print('Now trying neighbor ', k)
#         knn = KNeighborsClassifier(n_neighbors=k)
#         scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#         k_scores.append(scores.mean())
#     k_list = [i for i in k_range]

#     max_accuracy = max(k_scores)
#     neighbors_rough = []
#     for i, j in zip(k_list, k_scores):
#         if j == max_accuracy:
#             print('Fine tune max_accuracy occurs at neighbor #', i)   
#             neighbors_rough.append(i)
#     plt.plot(k_list, k_scores)
#     plt.xlabel('Value of neighbor k for KNN')
#     plt.ylabel('5-fold cross-validated accuracy')
#     plt.title(f'{enzyme} KNN accuracy with neighbors fine tune, best_neighbor = {neighbors_rough[0]}')

#     path = "knn_figures/"
#     isExist = os.path.exists(path) #printing if the path exists or not
#     print(path, ' folder is in directory: ', isExist)
#     if not isExist: # Create a new directory because it does not exist
#         os.makedirs(path)
#         print(path, " is created!")

#     plt.savefig(path + enzyme + '_fine_tune.png')

#     print('neighbors_fine for ', enzyme, ' : ', neighbors_rough)
#     print('max accuracy: ', max_accuracy)
#     return neighbors_rough[0]
# #####定义jak dataset处理原始数据
# # data_path = 'Data_MTATFP/data_label_0.25/'
# # for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
    
# #     data_partial = pd.read_csv(data_path + enzyme + '_partial.csv')
# #     data_test = pd.read_csv(data_path + enzyme + '_test.csv')
    
# #     data = JAK_dataset(data_partial['SMILES'], data_partial['Activity'])
# #     data_test_jak = JAK_dataset(data_test['SMILES'], data_test['Activity'])

# #     top_k = KNN_rough_tune(data.get_df(), enzyme)
# #     final_top_k = KNN_fine_tune(data.get_df(), enzyme, top_k)
# #     print(enzyme, ', best neighbor is ', final_top_k)
    
# #     knn = KNeighborsClassifier(n_neighbors=final_top_k)
# #     knn.fit(data.get_MACCS(), data.get_activity())
# #     knn.probability=True
# #     y_pred = knn.predict(data_test_jak.get_MACCS())
# #     y_prob = knn.predict_proba(data_test_jak.get_MACCS())
# #     print('for ', enzyme)
# #     evaluate(data_test_jak.get_activity(), y_pred, y_prob)
# #     filename = model_path + 'knn_' + enzyme + '.pkl'
# #     pickle.dump(knn, open(filename, 'wb'))
# data_path = 'processed_data/'  # Path to your processed data with MACCS fingerprints
# for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
#     # Load the MACCS data directly

#     data = pd.read_excel(data_path + enzyme + '_MACCS.xlsx', engine='openpyxl')
#         # Drop rows with NaN in 'Activity'
#     data = data.dropna(subset=['Activity'])

#     # Ensure required columns are present
#     header = ['bit' + str(i) for i in range(167)]
#     assert all(col in data.columns for col in header), f"MACCS fingerprint columns are missing for {enzyme}!"
#     assert 'Activity' in data.columns, f"Activity column is missing for {enzyme}!"

#     top_k = KNN_rough_tune(data, enzyme)
#     final_top_k = KNN_fine_tune(data, enzyme, top_k)
#     print(enzyme, ', best neighbor is ', final_top_k)

#     # Train the final model
#     X = data[header]
#     y = data['Activity']
#     knn = KNeighborsClassifier(n_neighbors=final_top_k)
#     knn.fit(X, y)

#     # Save the trained model
#     filename = model_path + 'knn_' + enzyme + '.pkl'
#     with open(filename, 'wb') as f:
#         pickle.dump(knn, f)
#     print(f'Model for {enzyme} saved to {filename}')



######划分训练集和测试集： 通过 train_test_split，首先将数据集分为 80% 的训练集和 20% 的测试集，然后将测试集进一步划分为 50% 的验证集和 50% 的最终测试集。
###交叉验证： 使用 cross_val_score 对训练集进行了 5 折交叉验证，评估模型的准确率。
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import warnings
from itertools import cycle  # 用于循环颜色和线型

warnings.filterwarnings("ignore", category=FutureWarning)

# 数据集划分和粗调函数
def KNN_rough_tune(data, enzyme):
    header = ['bit'+str(i) for i in range(167)]
    X = data[header]
    y = data['Activity']

    max_neighbors = len(X) - 1  # 样本数减去 1，避免包含自己作为邻居
    
    if max_neighbors < 50:
        print(f"Warning: {enzyme} 数据样本非常少，仅 {len(X)} 条记录，邻居数将动态调整。")
    
    # 确保k_range的最大值不会超过样本数
    k_range = range(1, min(500, max_neighbors) + 1)  # 取较小值，保证不超过最大样本数
    k_list = [item for item in k_range if item <= max_neighbors]

    print(f"Rough test neighbors for {enzyme} with {len(X)} samples (max neighbors = {max_neighbors}):")
    print(f"Testing k values: {k_list}")

    k_scores = []
    # Cross validation 5
    for k in k_list:
        if k % 50 == 0:
            print('Now trying neighbor#', k)
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    plt.plot(k_list, k_scores)
    plt.xlabel('Value of neighbor k for KNN')
    plt.ylabel('5-fold cross-validated accuracy')
    title = enzyme + ' KNN accuracy with neighbors ' + 'rough tune'
    plt.title(title)
    path = "knn_figures/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + enzyme + '_rough_tune.png')
    plt.close()

    max_accuracy = max(k_scores)
    neighbors_rough = []
    for i, j in zip(k_list, k_scores):
        if j == max_accuracy:
            print('Rough tune max_accuracy occurs at neighbor #', i)   
            neighbors_rough.append(i)
    
    return neighbors_rough[0]

# 细调 KNN 模型并输出评估结果
def KNN_fine_tune(data, enzyme='JAK1', rough_neighbor=10):
    header = ['bit'+str(i) for i in range(167)]
    X = data[header]
    y = data['Activity']

    print(f"################################################")
    print(f"FOR {enzyme}")
    print(f"Loaded data for {enzyme}: {X.shape}")
    print(f"Starting KNN training for {enzyme}...")

    # 数据集划分：80% 训练集，20% 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Step 1: Data split complete.")

    # 细调 k 值范围
    low_bound = max(rough_neighbor - 10, 1)
    up_bound = low_bound + 20 
    k_range = range(low_bound, up_bound)
    k_scores = []

    # 交叉验证：5折交叉验证
    for k in k_range:
        if k % 5 == 0:
            print(f'Now trying neighbor {k}')
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    # 网格搜索：调优超参数
    param_grid = {
        'n_neighbors': k_range
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    
    print(f"Best k found by GridSearchCV for {enzyme}: {best_k}")
    print(f"Best accuracy: {grid_search.best_score_}")

    # 绘制k值与准确度关系图
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of neighbor k for KNN')
    plt.ylabel('5-fold cross-validated accuracy')
    plt.title(f'{enzyme} KNN accuracy with neighbors fine tune, best_neighbor = {best_k}')
    
    path = "knn_figures/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + enzyme + '_fine_tune.png')
    
    # 使用调优后的 k 值训练最终模型
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    # 模型评估：预测测试集
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]  # 获取正类的概率值，用于 AUC 计算

    # 计算评估指标
    print(f'Evaluation Results for {enzyme}:')
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # 分类报告：精确度、召回率、F1 分数
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    # 保存最终模型
    model_path = "new_model/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    filename = model_path + 'KNN_' + enzyme + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(knn, f)
    
    print(f'Model saved to {filename}')
    print(f"################################################")
    
    return knn, X_test, y_test

# 绘制 ROC 曲线比较图（修改为统一风格）
def plot_roc_curve(models, X_test, y_test, enzyme_names, model_name="KNN"):
    plt.figure(figsize=(10, 8))
    
    # 定义颜色和线型循环器
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # 对每个模型计算 ROC 曲线并绘制
    for i, model in enumerate(models):
        # 获取每个模型的预测概率
        y_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
        fpr, tpr, _ = roc_curve(y_test, y_prob)  # 计算假阳性率和真阳性率
        roc_auc = auc(fpr, tpr)  # 计算AUC值
        
        plt.plot(fpr, tpr, 
                 color=next(colors),
                 linestyle=next(linestyles),
                 linewidth=2,
                 label=f'{enzyme_names[i]} (AUC = {roc_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {model_name} Model', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    os.makedirs('./roc_curves', exist_ok=True)
    plt.savefig(f'./roc_curves/{model_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    enzyme_names = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
    models = []
    test_data = []  # 存储每个酶的测试数据
    
    for enzyme in enzyme_names:
        # 加载数据并训练模型
        data = pd.read_excel(f'D:/CC/PycharmProjects/GoGT/JAK_ML/processed_data/NEW_{enzyme}_MACCS.xlsx', engine='openpyxl')
        data = data.dropna(subset=['Activity'])  # 删除 Activity 列为空的行
        
        # 先进行粗调
        rough_neighbor = KNN_rough_tune(data, enzyme)
        
        # 再进行细调
        knn, X_test, y_test = KNN_fine_tune(data, enzyme=enzyme, rough_neighbor=rough_neighbor)
        models.append(knn)
        test_data.append((X_test, y_test))
    
    # 为每个酶绘制单独的ROC曲线
    for i, enzyme in enumerate(enzyme_names):
        X_test, y_test = test_data[i]
        plot_roc_curve([models[i]], X_test, y_test, [enzyme], f"KNN_{enzyme}")
    
    # 绘制所有酶的综合ROC曲线
    # 注意：这里使用最后一个酶的测试数据，可能需要调整为合并所有测试数据
    # 此处为保持与原代码一致，使用最后一个酶的测试数据
    X_test, y_test = test_data[-1]
    plot_roc_curve(models, X_test, y_test, enzyme_names, "KNN")