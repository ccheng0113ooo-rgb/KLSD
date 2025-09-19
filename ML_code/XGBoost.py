
# import xgboost as xgb
# import pickle
# from tkinter import Grid
# from function import evaluate, load_data, save_tpr_fpr, load_tpr_fpr, get_tpr_fpr,get_preds
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# # 定义全局变量
# global header
# header = ['bit' + str(i) for i in range(167)]
# path = ''
# model_path = path + 'model/'

# # 循环处理每种酶
# for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
#     print(f"################################################")
#     print(f"Processing {enzyme}...")

#     # 加载数据
#     X, y = load_data(enzyme)
#     print(f"Loaded data for {enzyme}: {X.shape}")

#     # 检查并清理 NaN 数据
#     if y.isnull().sum() > 0:
#         print(f"Warning: Detected {y.isnull().sum()} NaN values in labels for {enzyme}. Cleaning data...")
#         data = pd.concat([X, y], axis=1)
#         data = data.dropna()  # 删除含有缺失值的行
#         X = data.iloc[:, :-1]
#         y = data.iloc[:, -1]

#     # 定义参数网格
#     params = {
#         'max_depth': [3, 6, 10],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [10, 50, 100, 500, 1000],
#         'colsample_bylevel': [0.3, 0.7, 1],
#         'colsample_bytree': [0.3, 0.7, 1]
#     }

#     # 网格搜索寻找最佳参数
#     model = xgb.XGBClassifier()
#     clf = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', verbose=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf.fit(X_train, y_train)
#     print('Best estimators:', clf.best_params_)

#     # 保存最佳参数
#     file = f'XGBoost_params/XGBoost_{enzyme}.pickle'
#     with open(file, 'wb') as f:
#         pickle.dump(clf.best_params_, f)
#     print('Best accuracy:', clf.best_score_)

#     # 加载最佳参数并重新训练模型
#     with open(file, 'rb') as f:
#         tuned_params = pickle.load(f)

#     model = xgb.XGBClassifier(**tuned_params)
#     model.fit(X_train, y_train)

#     # 在第一次划分后的测试集上再次划分验证集和测试集
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#     # 评估模型性能
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]
#     evaluate(y_test, y_pred, y_prob)
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))

#     # 保存模型
#     filename = f'XGBoost_{enzyme}.sav'
#     pickle.dump(model, open(model_path + filename, 'wb'))

#     # 保存 TPR 和 FPR
#     tpr_values, fpr_values = get_tpr_fpr(y_test, y_prob)
#     save_tpr_fpr('XGBoost', enzyme, tpr_values, fpr_values)

# print("All enzymes processed!")



#             #######ROC Curve Comparison for XGBoost
# import pickle
# from matplotlib import pyplot as plt

# # 加载多个文件的数据
# with open('AUC/XGBoost_JAK1_tpr.pickle', 'rb') as f:
#     tpr_xgboost_jak1 = pickle.load(f)
# with open('AUC/XGBoost_JAK1_fpr.pickle', 'rb') as f:
#     fpr_xgboost_jak1 = pickle.load(f)

# with open('AUC/XGBoost_JAK2_tpr.pickle', 'rb') as f:
#     tpr_xgboost_jak2 = pickle.load(f)
# with open('AUC/XGBoost_JAK2_fpr.pickle', 'rb') as f:
#     fpr_xgboost_jak2 = pickle.load(f)

# with open('AUC/XGBoost_JAK3_tpr.pickle', 'rb') as f:
#     tpr_xgboost_jak3 = pickle.load(f)
# with open('AUC/XGBoost_JAK3_fpr.pickle', 'rb') as f:
#     fpr_xgboost_jak3 = pickle.load(f)

# with open('AUC/XGBoost_TYK2_tpr.pickle', 'rb') as f:
#     tpr_xgboost_tyk2 = pickle.load(f)
# with open('AUC/XGBoost_TYK2_fpr.pickle', 'rb') as f:
#     fpr_xgboost_tyk2 = pickle.load(f)
# # 绘制比较图
# plt.plot(fpr_xgboost_jak1, tpr_xgboost_jak1, label='XGBoost JAK1')
# plt.plot(fpr_xgboost_jak2, tpr_xgboost_jak2, label='XGBoost JAK2')
# plt.plot(fpr_xgboost_jak3, tpr_xgboost_jak3, label='XGBoost JAK3')
# plt.plot(fpr_xgboost_tyk2, tpr_xgboost_tyk2, label='XGBoost TYK2')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison for XGBoost')
# plt.legend()
# plt.show()


#划分训练集和测试集： 通过 train_test_split，首先将数据集分为 80% 的训练集和 20% 的测试集，然后将测试集进一步划分为 50% 的验证集和 50% 的最终测试集。
##交叉验证： 使用 cross_val_score 对训练集进行了 5 折交叉验证，评估模型的准确率。
import os
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# 定义全局变量
header = ['bit' + str(i) for i in range(167)]
model_path = r'D:\CC\PycharmProjects\GoGT\JAK_ML\new_model'
# AUC 曲线数据保存路径
auc_path = r'D:\CC\PycharmProjects\GoGT\roc_curves'
if not os.path.exists(auc_path):
    os.makedirs(auc_path)

# 加载数据函数
def load_data(enzyme):
    data_path = f'D:/CC/PycharmProjects/GoGT/JAK_ML/processed_data/NEW_{enzyme}_MACCS.xlsx'
    data = pd.read_excel(data_path, engine='openpyxl')
    X = data.iloc[:, :-1]  # 所有特征列
    y = data.iloc[:, -1]   # 标签列
    return X, y

# 保存 TPR 和 FPR 数据
def save_tpr_fpr(model_name, enzyme, tpr_values, fpr_values):
    tpr_file = os.path.join(auc_path, f"{model_name}_{enzyme}_tpr.pickle")
    fpr_file = os.path.join(auc_path, f"{model_name}_{enzyme}_fpr.pickle")
    with open(tpr_file, 'wb') as tpr_f:
        pickle.dump(tpr_values, tpr_f)
    with open(fpr_file, 'wb') as fpr_f:
        pickle.dump(fpr_values, fpr_f)
    print(f"TPR and FPR saved for {model_name} on {enzyme}")

# 计算 TPR 和 FPR
def get_tpr_fpr(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return tpr, fpr

# 评估函数
def evaluate(y_true, y_pred, y_prob):
    print("Evaluation Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))

# 绘制 ROC 曲线（修改为统一风格）
def plot_roc_curve(enzymes, tpr_fpr_dict, model_name="XGBoost"):
    plt.figure(figsize=(10, 8))
    # 定义颜色和线型循环器
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    for enzyme in enzymes:
        if enzyme in tpr_fpr_dict:
            tpr, fpr = tpr_fpr_dict[enzyme]
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                     color=next(colors),
                     linestyle=next(linestyles),
                     linewidth=2,
                     label=f'{enzyme} (AUC = {roc_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {model_name} Model', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs('./roc_curves', exist_ok=True)
    plt.savefig(f'./roc_curves/{model_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
    tpr_fpr_dict = {}  # 用于存储每个酶的 TPR 和 FPR

    for enzyme in enzymes:
        print(f"################################################")
        print(f"Processing {enzyme}...")

        # 加载数据
        X, y = load_data(enzyme)
        print(f"Loaded data for {enzyme}: {X.shape}")

        # 检查并清理 NaN 数据
        if y.isnull().sum() > 0:
            print(f"Warning: Detected {y.isnull().sum()} NaN values in labels for {enzyme}. Cleaning data...")
            data = pd.concat([X, y], axis=1)
            data = data.dropna()  # 删除含有缺失值的行
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

        # 定义参数网格
        params = {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100, 500, 1000],
            'colsample_bylevel': [0.3, 0.7, 1],
            'colsample_bytree': [0.3, 0.7, 1]
        }

        # 网格搜索寻找最佳参数
        model = xgb.XGBClassifier()
        clf = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', verbose=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        print('Best estimators:', clf.best_params_)

        # 保存最佳参数
        best_params_file = os.path.join(auc_path, f'XGBoost_{enzyme}_best_params.pickle')
        with open(best_params_file, 'wb') as f:
            pickle.dump(clf.best_params_, f)
        print('Best accuracy:', clf.best_score_)

        # 加载最佳参数并重新训练模型
        with open(best_params_file, 'rb') as f:
            tuned_params = pickle.load(f)

        model = xgb.XGBClassifier(**tuned_params)
        model.fit(X_train, y_train)

        # 评估模型性能
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        evaluate(y_test, y_pred, y_prob)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # 保存模型
        filename = f'XGBoost_{enzyme}.sav'
        pickle.dump(model, open(os.path.join(model_path, filename), 'wb'))

        # 保存 TPR 和 FPR
        tpr_values, fpr_values = get_tpr_fpr(y_test, y_prob)
        save_tpr_fpr('XGBoost', enzyme, tpr_values, fpr_values)
        tpr_fpr_dict[enzyme] = (tpr_values, fpr_values)

    # 绘制所有 ROC 曲线（调用修改后的绘图函数）
    plot_roc_curve(enzymes, tpr_fpr_dict, "XGBoost")
    print("All enzymes processed!")


    