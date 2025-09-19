
# from sklearn.svm import SVC
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
# from function import evaluate,get_preds,save_tpr_fpr
# import pickle

# global header
# header = ['bit'+str(i) for i in range(167)]

# def SVM_batch(enzyme):
#     path = 'processed_data/' + enzyme + '_' + 'MACCS.xlsx'
#     # data = pd.read_csv(path)
#     try:
#         # 使用 read_excel 读取文件
#         data = pd.read_excel(path, engine='openpyxl')
#     except Exception as e:
#         print(f"读取文件 {path} 时出错: {e}")
#         return
#         # 检查并移除 NaN
#     data = data.dropna(subset=['Activity'])  # 移除 Activity 列中含有 NaN 的行

   
#     header = ['bit'+str(i) for i in range(167)]
#     X = data[header]
#     y = data['Activity']
#     for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#         if k =='poly':
#             model = SVC(kernel=k, degree=8)
#         else:
#             model = SVC(kernel=k)
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, 
#         random_state=42, test_size = 0.2)
#         scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
#         print('SVM: ',k, 'Scores: ', scores.mean())
#         model.probability=True
#         model.fit(X_train, y_train)
#     #     y_pred = model.predict(X_test)
#         X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 
#         random_state=42, test_size = 0.5)
#         y_prob = model.predict_proba(X_test)
#         proba = y_prob[:, 1]
#         y_pred = get_preds(0.5, proba)
#         evaluate(y_test, y_pred, y_prob)

#         # Get AUC values for AUC figure
#         roc_values = []
#         for thresh in np.linspace(0, 1, 100):
#             preds = get_preds(thresh, proba)
#             tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
#             tpr = tp/(tp+fn)
#             fpr = fp/(fp+tn)
#             roc_values.append([tpr, fpr])
#         tpr_values, fpr_values = zip(*roc_values)
        
#         save_tpr_fpr('SVM_'+k, enzyme, tpr_values, fpr_values)

#         filename = 'model/SVM_'+k+'_'+enzyme+'.sav'
#         pickle.dump(model, open(filename, 'wb'))

# enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
# for enzyme in enzymes: 
#     print('################################################')
#     print('FOR', enzyme)
#     SVM_batch(enzyme)




# #######ROC Curve Comparison for JAK1
# import pickle
# from matplotlib import pyplot as plt

# # 加载多个文件的数据
# with open('AUC/SVM_linear_JAK1_tpr.pickle', 'rb') as f:
#     tpr_linear = pickle.load(f)
# with open('AUC/SVM_linear_JAK1_fpr.pickle', 'rb') as f:
#     fpr_linear = pickle.load(f)

# with open('AUC/SVM_poly_JAK1_tpr.pickle', 'rb') as f:
#     tpr_poly = pickle.load(f)
# with open('AUC/SVM_poly_JAK1_fpr.pickle', 'rb') as f:
#     fpr_poly = pickle.load(f)

# with open('AUC/SVM_rbf_JAK1_tpr.pickle', 'rb') as f:
#     tpr_rbf = pickle.load(f)
# with open('AUC/SVM_rbf_JAK1_fpr.pickle', 'rb') as f:
#     fpr_rbf = pickle.load(f)

# with open('AUC/SVM_sigmoid_JAK1_tpr.pickle', 'rb') as f:
#     tpr_sigmoid = pickle.load(f)
# with open('AUC/SVM_sigmoid_JAK1_fpr.pickle', 'rb') as f:
#     fpr_sigmoid = pickle.load(f)
# # 绘制比较图
# plt.plot(fpr_linear, tpr_linear, label='SVM Linear')
# plt.plot(fpr_poly, tpr_poly, label='SVM Poly')
# plt.plot(fpr_rbf, tpr_rbf, label='SVM RBF')
# plt.plot(fpr_sigmoid, tpr_sigmoid, label='SVM Sigmoid')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison for JAK1')
# plt.legend()
# plt.show()
# #####ROC Curve Comparison for TYK2
# import pickle
# from matplotlib import pyplot as plt

# # 加载多个文件的数据
# with open('AUC/SVM_linear_TYK2_tpr.pickle', 'rb') as f:
#     tpr_linear = pickle.load(f)
# with open('AUC/SVM_linear_TYK2_fpr.pickle', 'rb') as f:
#     fpr_linear = pickle.load(f)

# with open('AUC/SVM_poly_TYK2_tpr.pickle', 'rb') as f:
#     tpr_poly = pickle.load(f)
# with open('AUC/SVM_poly_TYK2_fpr.pickle', 'rb') as f:
#     fpr_poly = pickle.load(f)

# with open('AUC/SVM_rbf_TYK2_tpr.pickle', 'rb') as f:
#     tpr_rbf = pickle.load(f)
# with open('AUC/SVM_rbf_TYK2_fpr.pickle', 'rb') as f:
#     fpr_rbf = pickle.load(f)

# with open('AUC/SVM_sigmoid_TYK2_tpr.pickle', 'rb') as f:
#     tpr_sigmoid = pickle.load(f)
# with open('AUC/SVM_sigmoid_TYK2_fpr.pickle', 'rb') as f:
#     fpr_sigmoid = pickle.load(f)
# # 绘制比较图
# plt.plot(fpr_linear, tpr_linear, label='SVM Linear')
# plt.plot(fpr_poly, tpr_poly, label='SVM Poly')
# plt.plot(fpr_rbf, tpr_rbf, label='SVM RBF')
# plt.plot(fpr_sigmoid, tpr_sigmoid, label='SVM Sigmoid')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison for TYK2')
# plt.legend()
# plt.show()


# #######ROC Curve Comparison for JAK2
# import pickle
# from matplotlib import pyplot as plt

# # 加载多个文件的数据
# with open('AUC/SVM_linear_JAK2_tpr.pickle', 'rb') as f:
#     tpr_linear = pickle.load(f)
# with open('AUC/SVM_linear_JAK2_fpr.pickle', 'rb') as f:
#     fpr_linear = pickle.load(f)

# with open('AUC/SVM_poly_JAK2_tpr.pickle', 'rb') as f:
#     tpr_poly = pickle.load(f)
# with open('AUC/SVM_poly_JAK2_fpr.pickle', 'rb') as f:
#     fpr_poly = pickle.load(f)

# with open('AUC/SVM_rbf_JAK2_tpr.pickle', 'rb') as f:
#     tpr_rbf = pickle.load(f)
# with open('AUC/SVM_rbf_JAK2_fpr.pickle', 'rb') as f:
#     fpr_rbf = pickle.load(f)

# with open('AUC/SVM_sigmoid_JAK2_tpr.pickle', 'rb') as f:
#     tpr_sigmoid = pickle.load(f)
# with open('AUC/SVM_sigmoid_JAK2_fpr.pickle', 'rb') as f:
#     fpr_sigmoid = pickle.load(f)
# # 绘制比较图
# plt.plot(fpr_linear, tpr_linear, label='SVM Linear')
# plt.plot(fpr_poly, tpr_poly, label='SVM Poly')
# plt.plot(fpr_rbf, tpr_rbf, label='SVM RBF')
# plt.plot(fpr_sigmoid, tpr_sigmoid, label='SVM Sigmoid')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison for JAK2')
# plt.legend()
# plt.show()


# #######ROC Curve Comparison for JAK3
# import pickle
# from matplotlib import pyplot as plt

# # 加载多个文件的数据
# with open('AUC/SVM_linear_JAK3_tpr.pickle', 'rb') as f:
#     tpr_linear = pickle.load(f)
# with open('AUC/SVM_linear_JAK3_fpr.pickle', 'rb') as f:
#     fpr_linear = pickle.load(f)

# with open('AUC/SVM_poly_JAK3_tpr.pickle', 'rb') as f:
#     tpr_poly = pickle.load(f)
# with open('AUC/SVM_poly_JAK3_fpr.pickle', 'rb') as f:
#     fpr_poly = pickle.load(f)

# with open('AUC/SVM_rbf_JAK3_tpr.pickle', 'rb') as f:
#     tpr_rbf = pickle.load(f)
# with open('AUC/SVM_rbf_JAK3_fpr.pickle', 'rb') as f:
#     fpr_rbf = pickle.load(f)

# with open('AUC/SVM_sigmoid_JAK3_tpr.pickle', 'rb') as f:
#     tpr_sigmoid = pickle.load(f)
# with open('AUC/SVM_sigmoid_JAK3_fpr.pickle', 'rb') as f:
#     fpr_sigmoid = pickle.load(f)
# # 绘制比较图
# plt.plot(fpr_linear, tpr_linear, label='SVM Linear')
# plt.plot(fpr_poly, tpr_poly, label='SVM Poly')
# plt.plot(fpr_rbf, tpr_rbf, label='SVM RBF')
# plt.plot(fpr_sigmoid, tpr_sigmoid, label='SVM Sigmoid')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison for JAK3')
# plt.legend()
# plt.show()



#划分训练集和测试集： 通过 train_test_split，首先将数据集分为 80% 的训练集和 20% 的测试集，然后将测试集进一步划分为 50% 的验证集和 50% 的最终测试集。
##交叉验证： 使用 cross_val_score 对训练集进行了 5 折交叉验证，评估模型的准确率。
from sklearn.svm import SVC
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
auc_path = r'D:\CC\PycharmProjects\GoGT\JAK_ML\new_AUC'
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
def plot_roc_curve(enzymes, tpr_fpr_dict, model_name="SVM"):
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

# SVM 单模型训练和评估函数
def SVM_single(X, y, enzyme):
    print(f"Starting SVM training for {enzyme}...")
    
    # 数据集划分：80%训练集，20%测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Step 1: Data split complete.")
    
    # # 测试集进一步划分：50%验证集，50%最终测试集
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    # print("Step 2: Test set split complete.")
    
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.01, 0.1, 1]
    }
    
    # 网格搜索寻找最佳参数
    model = SVC(probability=True)
    clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    clf.fit(X_train, y_train)
    
    # 输出最佳参数和得分
    print('Best estimators:', clf.best_params_)
    print('Best accuracy:', clf.best_score_)
    
    # 使用最佳参数重新训练模型
    tuned_model = SVC(**clf.best_params_, probability=True)
    tuned_model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)
    
    # 模型评估
    evaluate(y_test, y_pred, y_prob)
    
    # 保存 TPR 和 FPR
    tpr_values, fpr_values = get_tpr_fpr(y_test, y_prob)
    save_tpr_fpr('SVM', enzyme, tpr_values, fpr_values)
    
    # 保存模型
    filename = f'SVM_{enzyme}.sav'
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
            
            # 训练 SVM 模型并获取 TPR 和 FPR
            tpr_values, fpr_values = SVM_single(X, y, enzyme)
            tpr_fpr_dict[enzyme] = (tpr_values, fpr_values)
        
        except FileNotFoundError:
            print(f"Error: Data file for {enzyme} not found. Ensure the file exists at {data_path}.")
        except Exception as e:
            print(f"An error occurred while processing {enzyme}: {e}")
    
    # 绘制所有 ROC 曲线（修改调用方式）
    plot_roc_curve(enzymes, tpr_fpr_dict, "SVM")