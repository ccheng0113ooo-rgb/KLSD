##原始的没有进行网格搜索的代码
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn import metrics
# import pickle
# import os
# import pandas as pd

# # 确保模型保存路径存在
# model_path = 'model/'
# if not os.path.exists(model_path):
#     os.makedirs(model_path)

# # AUC 曲线数据保存路径
# auc_path = 'AUC/'
# if not os.path.exists(auc_path):
#     os.makedirs(auc_path)

# # 简化评估函数
# def evaluate(y_true, y_pred, y_prob):
#     print("Evaluation Results:")
#     print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
#     print("ROC AUC:", metrics.roc_auc_score(y_true, y_prob[:, 1]))

# # 保存 TPR 和 FPR 数据
# def save_tpr_fpr(model_name, enzyme, tpr_values, fpr_values):
#     tpr_file = os.path.join(auc_path, f"{model_name}_{enzyme}_tpr.pickle")
#     fpr_file = os.path.join(auc_path, f"{model_name}_{enzyme}_fpr.pickle")
#     with open(tpr_file, 'wb') as tpr_f:
#         pickle.dump(tpr_values, tpr_f)
#     with open(fpr_file, 'wb') as fpr_f:
#         pickle.dump(fpr_values, fpr_f)
#     print(f"TPR and FPR saved for {model_name} on {enzyme}")

# # 计算 TPR 和 FPR
# def get_tpr_fpr(y_true, y_prob):
#     fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:, 1])
#     return tpr, fpr

# # RF 单模型训练和评估函数
# def RF_single(X, y, n_estimator, enzyme, test_size):
#     print(f"Starting RF training for {enzyme}...")
    
#     # 数据集划分：80%训练集，20%测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     print("Step 1: Data split complete.")
    
#     # 测试集进一步划分：50%验证集，50%最终测试集
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#     print("Step 2: Test set split complete.")
    
#     # 模型初始化和训练
#     RF = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
    
#     # 交叉验证：5折交叉验证
#     cross_val_scores = cross_val_score(RF, X_train, y_train, cv=5, scoring='accuracy')
#     print("Step 3: Cross-validation complete. Mean CV score:", cross_val_scores.mean())
    
#     # 训练模型
#     RF.fit(X_train, y_train)
#     print("Step 4: Model training complete.")
    
#     # 模型预测
#     y_pred = RF.predict(X_test)
#     y_prob = RF.predict_proba(X_test)
#     print("Step 5: Prediction complete.")
    
#     # 模型评估
#     evaluate(y_test, y_pred, y_prob)
    
#     # 保存 TPR 和 FPR
#     tpr_values, fpr_values = get_tpr_fpr(y_test, y_prob)
#     save_tpr_fpr('RF', enzyme, tpr_values, fpr_values)
    
#     # 保存模型
#     filename = f'RF_{enzyme}.sav'
#     modelname = os.path.join(model_path, filename)
#     pickle.dump(RF, open(modelname, 'wb'))
#     print(f"Model saved to {modelname}")

# # 主程序
# if __name__ == "__main__":
#     # 加载数据
#     enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
#     for enzyme in enzymes:
#         print("################################################")
#         print(f"FOR {enzyme}")
        
#         try:
#             # 读取数据
#             data_path = f'processed_data/{enzyme}_MACCS.xlsx'
#             data = pd.read_excel(data_path, engine='openpyxl')
#             print(f"Loaded data for {enzyme}: {data.shape}")
            
#             # 删除含缺失值的行
#             data = data.dropna()  # 删除缺失值

#             # 提取特征和标签
#             X = data.iloc[:, :-1]  # 所有特征列
#             y = data.iloc[:, -1]   # 标签列
            
#             # 检查是否有缺失值
#             if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
#                 print(f"Warning: Missing values detected in {enzyme} data. Please clean the dataset.")
#                 continue
            
#             # 检查标签是否只有 0 和 1
#             if not set(y.unique()).issubset({0, 1}):
#                 print(f"Error: Invalid labels detected in {enzyme} data. Ensure labels are 0 or 1.")
#                 continue
            
#             # 训练随机森林模型
#             RF_single(X, y, n_estimator=100, enzyme=enzyme, test_size=0.2)
        
#         except FileNotFoundError:
#             print(f"Error: Data file for {enzyme} not found. Ensure the file exists at {data_path}.")
#         except Exception as e:
#             print(f"An error occurred while processing {enzyme}: {e}")



#             #######ROC Curve Comparison for RF
# import pickle
# from matplotlib import pyplot as plt

# # 加载多个文件的数据
# with open('AUC/RF_JAK1_tpr.pickle', 'rb') as f:
#     tpr_rf_jak1 = pickle.load(f)
# with open('AUC/RF_JAK1_fpr.pickle', 'rb') as f:
#     fpr_rf_jak1 = pickle.load(f)

# with open('AUC/RF_JAK2_tpr.pickle', 'rb') as f:
#     tpr_rf_jak2 = pickle.load(f)
# with open('AUC/RF_JAK2_fpr.pickle', 'rb') as f:
#     fpr_rf_jak2 = pickle.load(f)

# with open('AUC/RF_JAK3_tpr.pickle', 'rb') as f:
#     tpr_rf_jak3 = pickle.load(f)
# with open('AUC/RF_JAK3_fpr.pickle', 'rb') as f:
#     fpr_rf_jak3 = pickle.load(f)

# with open('AUC/RF_TYK2_tpr.pickle', 'rb') as f:
#     tpr_rf_tyk2 = pickle.load(f)
# with open('AUC/RF_TYK2_fpr.pickle', 'rb') as f:
#     fpr_rf_tyk2 = pickle.load(f)
# # 绘制比较图
# plt.plot(fpr_rf_jak1, tpr_rf_jak1, label='RF JAK1')
# plt.plot(fpr_rf_jak2, tpr_rf_jak2, label='RF JAK2')
# plt.plot(fpr_rf_jak3, tpr_rf_jak3, label='RF JAK3')
# plt.plot(fpr_rf_tyk2, tpr_rf_tyk2, label='RF TYK2')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison for RF')
# plt.legend()
# plt.show()



####划分训练集和测试集： 通过 train_test_split，首先将数据集分为 80% 的训练集和 20% 的测试集，然后将测试集进一步划分为 50% 的验证集和 50% 的最终测试集。
###交叉验证： 使用 cross_val_score 对训练集进行了 5 折交叉验证，评估模型的准确率。
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle  # 用于循环颜色和线型

# 确保模型保存路径存在
model_path = r'D:\CC\PycharmProjects\GoGT\JAK_ML\new_model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# AUC 曲线数据保存路径
auc_path = r'D:\CC\PycharmProjects\GoGT\roc_curves'
if not os.path.exists(auc_path):
    os.makedirs(auc_path)

# 评估函数
def evaluate(y_true, y_pred, y_prob):
    print("Evaluation Results:")
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
    print("Precision:", metrics.precision_score(y_true, y_pred))
    print("Recall:", metrics.recall_score(y_true, y_pred))
    print("F1 Score:", metrics.f1_score(y_true, y_pred))
    print("ROC AUC:", metrics.roc_auc_score(y_true, y_prob[:, 1]))

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
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:, 1])
    return tpr, fpr

# 绘制 ROC 曲线（修改为新样式）
def plot_roc_curve(enzymes, tpr_fpr_dict, model_name="RF"):
    plt.figure(figsize=(10, 8))
    
    # 定义颜色和线型循环器
    colors = cycle(['darkorange', 'blue', 'green', 'red'])
    linestyles = cycle(['-', '--', '-.', ':'])
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    for enzyme in enzymes:
        if enzyme in tpr_fpr_dict:
            tpr, fpr = tpr_fpr_dict[enzyme]
            roc_auc = metrics.auc(fpr, tpr)
            
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

# RF 单模型训练和评估函数
def RF_single(X, y, enzyme):
    print(f"Starting RF training for {enzyme}...")
    
    # 数据集划分：80%训练集，20%测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Step 1: Data split complete.")
    
    # # 测试集进一步划分：50%验证集，50%最终测试集
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    # print("Step 2: Test set split complete.")
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 6, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 网格搜索寻找最佳参数
    model = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    clf.fit(X_train, y_train)
    
    # 输出最佳参数和得分
    print('Best estimators:', clf.best_params_)
    print('Best accuracy:', clf.best_score_)
    
    # 使用最佳参数重新训练模型
    tuned_model = RandomForestClassifier(**clf.best_params_, random_state=42)
    tuned_model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)
    
    # 模型评估
    evaluate(y_test, y_pred, y_prob)
    
    # 保存 TPR 和 FPR
    tpr_values, fpr_values = get_tpr_fpr(y_test, y_prob)
    save_tpr_fpr('RF', enzyme, tpr_values, fpr_values)
    
    # 保存模型
    filename = f'RF_{enzyme}.sav'
    pickle.dump(tuned_model, open(os.path.join(model_path, filename), 'wb'))
    print(f"Model saved to {os.path.join(model_path, filename)}")
    
    return tpr_values, fpr_values

# 主程序
if __name__ == "__main__":
    # 加载数据
    enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
    tpr_fpr_dict = {}  # 用于存储每个酶的 TPR 和 FPR
    
    for enzyme in enzymes:
        print("################################################")
        print(f"FOR {enzyme}")
        
        try:
            # 读取数据
            data_path = f'D:/CC/PycharmProjects/GoGT/JAK_ML/processed_data/NEW_{enzyme}_MACCS.xlsx'
            data = pd.read_excel(data_path, engine='openpyxl')
            print(f"Loaded data for {enzyme}: {data.shape}")
            
            # 删除含缺失值的行
            data = data.dropna()  # 删除缺失值

            # 提取特征和标签
            X = data.iloc[:, :-1]  # 所有特征列
            y = data.iloc[:, -1]   # 标签列
            
            # 检查标签是否只有 0 和 1
            if not set(y.unique()).issubset({0, 1}):
                print(f"Error: Invalid labels detected in {enzyme} data. Ensure labels are 0 or 1.")
                continue
            
            # 训练 RF 模型并获取 TPR 和 FPR
            tpr_values, fpr_values = RF_single(X, y, enzyme)
            tpr_fpr_dict[enzyme] = (tpr_values, fpr_values)
        
        except FileNotFoundError:
            print(f"Error: Data file for {enzyme} not found. Ensure the file exists at {data_path}.")
        except Exception as e:
            print(f"An error occurred while processing {enzyme}: {e}")
    
    # 绘制所有 ROC 曲线
    plot_roc_curve(enzymes, tpr_fpr_dict, "RandomForest")
    print("ROC curves plotted successfully!")